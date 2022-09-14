""" Preprocessor Module that given a filelist containing text, wav, textgrid:
    - extracts log Mel spectral features
    - extracts f0 (phone-level or frame-level)
    - extracts durations
    - extracts energy
    - extracts inputs (ex. phonological feats)
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyworld as pw
import torch  # fix torch imports
import torchaudio.functional as F
import torchaudio.transforms as T
from loguru import logger
from tabulate import tabulate
from torch import Tensor, linalg, mean, tensor
from torchaudio import load as load_audio
from torchaudio import save as save_audio
from torchaudio.sox_effects import apply_effects_tensor
from tqdm import tqdm

from config import ConfigError
from text import TextProcessor
from utils import read_textgrid


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.missing_files: List[str] = []
        self.audio_config = config["preprocessing"]["audio"]
        self.text_processor = TextProcessor(config)
        self.feature_prediction_filelist = self.config["training"][
            "feature_prediction"
        ]["filelist_loader"](self.config["training"]["feature_prediction"]["filelist"])
        self.vocoder_filelist = self.config["training"]["vocoder"]["filelist_loader"](
            self.config["training"]["vocoder"]["filelist"]
        )
        self.data_dir = Path(self.config["preprocessing"]["data_dir"])
        self.save_dir = Path(self.config["preprocessing"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # Define Spectral Transform
        # Gah, so many ways to do this: https://github.com/CookiePPP/VocoderComparisons/issues/3
        if self.audio_config["spec_type"] == "mel":
            self.spectral_transform = T.MelSpectrogram(
                sample_rate=self.audio_config["target_sampling_rate"],
                n_fft=self.audio_config["n_fft"],
                win_length=self.audio_config["fft_window_frames"],
                hop_length=self.audio_config["fft_hop_frames"],
                n_mels=self.audio_config["n_mels"],
            )
        elif config["preprocessing"]["audio"]["spec_type"] == "linear":
            self.spectral_transform = T.Spectrogram(
                n_fft=self.audio_config["n_fft"],
                win_length=self.audio_config["fft_window_frames"],
                hop_length=self.audio_config["fft_hop_frames"],
            )
        elif config["preprocessing"]["audio"]["spec_type"] == "raw":
            self.spectral_transform = T.Spectrogram(
                n_fft=self.audio_config["n_fft"],
                win_length=self.audio_config["fft_window_frames"],
                hop_length=self.audio_config["fft_hop_frames"],
                power=None,
            )
        else:
            raise ConfigError(
                f"Spectral feature specification {config['preprocessing']['audio']['spec_type'] == 'mel'} is not supported. Please edit your config file."
            )

    def process_audio_for_alignment(self, wav_path) -> Tuple[Tensor, int]:
        """Process audio with any Sox Effects

        Args:
            wav_path (str): path to wav file

        Returns:
            [Tensor, int]: audio Tensor, sampling rate
        """
        audio, sampling_rate = load_audio(wav_path)
        if self.config["preprocessing"]["audio"]["sox_effects"]:
            audio = apply_effects_tensor(
                audio,
                sampling_rate,
                self.config["preprocessing"]["audio"]["sox_effects"],
            )
        return audio

    def process_audio(self, wav_path: str) -> Tensor:
        """Process audio

        Args:
            wav_path (str): path to wav file
        Returns:
            [Tensor, int]: audio Tensor, sampling rate
        """

        return load_audio(wav_path)

    def extract_spectral_features(self, audio_tensor: Tensor):
        """Given an audio tensor, extract the log Mel spectral features
        from the given start and end points

        Args:
            audio_tensor (Tensor): Tensor trimmed according
        """
        return self.spectral_transform(audio_tensor)

    def extract_f0(self, audio_tensor: Tensor):
        """Given an audio tensor, extract the f0

        TODO: consider CWT and Parselmouth

        Comparison with other implementations:
            - ming024 & Christoph Minxhoffer use the pyworld implementation and interpolate along with phone averaging
            - the Lightspeech implementation seems to use pyworld implementation and not interpolate or average
            - Christoph Minxhoffer reported no significant differences with continuous wavelet transform so it is not implemented here

        Args:
            spectral_feature_tensor (Tensor): tensor of spectral features extracted from audio
        """
        if self.config["preprocessing"]["f0_type"] == "pyworld":
            pitch, t = pw.dio(
                audio_tensor.squeeze(0)
                .numpy()
                .astype(
                    np.float64
                ),  # TODO: why are these np.float64, maybe it's just what pw expects?
                self.audio_config["target_sampling_rate"],
                frame_period=self.audio_config["fft_hop_frames"]
                / self.audio_config["target_sampling_rate"]
                * 1000,
                speed=4,
            )
            pitch = pw.stonemask(
                audio_tensor.squeeze(0).numpy().astype(np.float64),
                pitch,
                t,
                self.audio_config["target_sampling_rate"],
            )
            pitch = tensor(pitch)
            # TODO: consider interpolating by default when using PyWorld pitch detection
        elif self.config["preprocessing"]["f0_type"] == "kaldi":
            pitch = F.compute_kaldi_pitch(
                waveform=audio_tensor,
                sample_rate=self.audio_config["target_sampling_rate"],
                frame_length=self.audio_config["fft_window_frames"]
                / self.audio_config["target_sampling_rate"]
                * 1000,
                frame_shift=self.audio_config["fft_hop_frames"]
                / self.audio_config["target_sampling_rate"]
                * 1000,
                min_f0=50,
                max_f0=400,
            )[0][
                ..., 1
            ]  # TODO: the docs and C Minxhoffer implementation take [..., 0] but this doesn't appear to be the pitch, at least for this version of torchaudio.
        else:
            raise ConfigError(
                f"Sorry, the f0 estimation type '{self.config['preprocessing']['f0_type']}' is not supported. Please edit your config file."
            )
        return pitch

    def extract_durations(self, textgrid_path: str):
        """Extract durations from a textgrid path
           Don't use tgt package because it ignores silence

        Args:
            textgrid_path (str): path to a textgrid file
        """
        tg = read_textgrid(textgrid_path)
        phones = tg.get_tier("phones")
        return [
            {
                "start": x[0],
                "end": x[1],
                "dur_ms": (x[1] - x[0]) * 1000,
                "dur_frames": int(
                    (
                        np.round(
                            x[1]
                            * self.audio_config["target_sampling_rate"]
                            / self.audio_config["fft_hop_frames"]
                        )
                        - np.round(
                            x[0]
                            * self.audio_config["target_sampling_rate"]
                            / self.audio_config["fft_hop_frames"]
                        )
                    )
                ),
                "phone": x[2],
            }
            for x in phones.get_all_intervals()
        ]

    def average_data_by_durations(self, data, durations):
        current_frame_position = 0
        new_data = []
        for duration in durations:
            if duration["dur_frames"] > 0:
                new_data.append(
                    mean(
                        data[
                            current_frame_position : current_frame_position
                            + duration["dur_frames"]
                        ]
                    )
                )
            else:
                new_data.append(1e-7)
            current_frame_position += duration["dur_frames"]
        return tensor(new_data)

    def extract_energy(self, spectral_feature_tensor: Tensor):
        """Given a spectral feature tensor, and durations extract the energy averaged across a phone

        Args:
            spectral_feature_tensor (Tensor): tensor of spectral features extracted from audio
            durations (_type_): _descriptiont    #TODO
        """
        energy = linalg.norm(spectral_feature_tensor, dim=1)
        return energy

    def extract_text_inputs(self, text, use_pfs=False):
        """Given some text, normalize it, g2p it, and save as one-hot or multi-hot phonological feature vectors

        Args:
            text (str): text
        """
        if use_pfs:
            return self.text_processor.text_to_phonological_features(text)
        else:
            return self.text_processor.text_to_sequence(text)

    def collect_files(self, filelist):
        for f in filelist:
            audio_path = self.data_dir / (f["basename"] + ".wav")
            if not audio_path.exists():
                logger.warning(f"File '{f}' if missing and will not be processed.")
                self.missing_files.append(f)
            else:
                yield f

    def report(self, processed, tablefmt="simple"):
        headers = ["type", "quantity"]
        table = [
            ["missing files", len(self.missing_files)],
            ["missing symbols", len(self.text_processor.missing_symbols)],
            ["duplicate symbols", len(self.text_processor.duplicate_symbols)],
        ]
        return tabulate(table, headers, tablefmt=tablefmt)

    def preprocess(
        self,
        filelist,
        process_audio=False,
        process_sox_audio=False,
        process_spec=False,
        process_energy=False,
        process_f0=False,
        process_duration=False,
        process_pfs=False,
        process_text=False,
    ):
        # TODO: use multiprocessing
        processed = 0
        for f in tqdm(self.collect_files(filelist), total=len(filelist)):
            speaker = "default" if "speaker" not in f else f["speaker"]
            language = "default" if "language" not in f else f["language"]
            audio = None
            spec = None
            if process_text:
                torch.save(
                    self.extract_text_inputs(f["text"]),
                    self.save_dir / f"{f['basename']}-{speaker}-{language}-text.npy",
                )
            if process_pfs:
                torch.save(
                    self.extract_text_inputs(f["text"], use_pfs=True),
                    self.save_dir / f"{f['basename']}-{speaker}-{language}-pfs.npy",
                )
            if process_sox_audio:
                save_audio(
                    self.save_dir / f"{f['basename']}-{speaker}-{language}-pfs.npy",
                    self.process_audio_for_alignment(
                        self.data_dir / (f["basename"] + ".wav")
                    ),
                    self.config["preprocessing"]["audio"]["alignment_sampling_rate"],
                    encoding="PCM_S",
                    bits_per_sample=self.config["preprocessing"]["audio"][
                        "alignment_bit_depth"
                    ],
                )
            if process_audio:
                audio, _ = self.process_audio(self.data_dir / (f["basename"] + ".wav"))
                torch.save(
                    audio,
                    self.save_dir / f"{f['basename']}-{speaker}-{language}-audio.npy",
                )
            if process_spec:
                if audio is None:
                    audio, _ = self.process_audio(
                        self.data_dir / (f["basename"] + ".wav")
                    )
                spec = self.extract_spectral_features(audio)
                torch.save(
                    spec,
                    self.save_dir
                    / f"{f['basename']}-{speaker}-{language}-spec-{self.audio_config['spec_type']}.npy",
                )
            if process_f0:
                if audio is None:
                    audio, _ = self.process_audio(
                        self.data_dir / (f["basename"] + ".wav")
                    )
                torch.save(
                    self.extract_f0(audio),
                    self.save_dir / f"{f['basename']}-{speaker}-{language}-f0.npy",
                )
            if process_energy:
                if spec is None:
                    spec = self.extract_spectral_features(audio)
                torch.save(
                    self.extract_energy(spec),
                    self.save_dir / f"{f['basename']}-{speaker}-{language}-energy.npy",
                )
            if process_duration:
                dur_path = self.data_dir / f["basename"] + ".TextGrid"
                if not dur_path.exists():
                    logger.warning(f"File '{f}' if missing and will not be processed.")
                    self.missing_files.append(f)
                torch.save(
                    self.extract_durations(dur_path),
                    self.save_dir
                    / f"{f['basename']}-{speaker}-{language}-duration.npy",
                )
            processed += 1
        logger.info(self.report(processed))
