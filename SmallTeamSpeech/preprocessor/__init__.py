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
from loguru import logger
from tabulate import tabulate
from torch import Tensor, linalg, mean, tensor
from torchaudio import load as load_audio
from torchaudio import save as save_audio
from torchaudio.sox_effects import apply_effects_tensor
from tqdm import tqdm

from config import ConfigError
from text import TextProcessor
from utils import get_spectral_transform, read_textgrid, write_filelist


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.missing_files: List[str] = []
        self.skipped_processes = 0
        self.audio_config = config["preprocessing"]["audio"]
        self.sep = config["preprocessing"]["value_separator"]
        self.text_processor = TextProcessor(config)
        self.data_dir = Path(self.config["preprocessing"]["data_dir"])
        self.save_dir = Path(self.config["preprocessing"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filelist = self.config["preprocessing"]["filelist_loader"](
            self.config["preprocessing"]["filelist"]
        )
        # Define Spectral Transform
        # Gah, so many ways to do this: https://github.com/CookiePPP/VocoderComparisons/issues/3

        self.spectral_transform = get_spectral_transform(
            self.audio_config["spec_type"],
            self.audio_config["n_fft"],
            self.audio_config["fft_window_frames"],
            self.audio_config["fft_hop_frames"],
            sample_rate=self.audio_config["target_sampling_rate"],
            n_mels=self.audio_config["n_mels"],
            f_min=self.audio_config["f_min"],
            f_max=self.audio_config["f_max"],
        )

        if self.spectral_transform is None:
            raise ConfigError(
                f"Spectral feature specification {self.audio_config['spec_type']} is not supported. Please edit your config file."
            )

    def process_audio(
        self, wav_path: str, normalize=True, use_effects=False
    ) -> Tuple[Tensor, int]:
        """Process audio

        Args:
            wav_path (str): path to wav file
            normalize (bool): normalizes to float32, NOT volume normalization
        Returns:
            [Tensor, int]: audio Tensor, sampling rate
        """

        audio, sr = load_audio(wav_path, normalize=normalize)
        if use_effects and self.config["preprocessing"]["audio"]["sox_effects"]:
            audio, sr = apply_effects_tensor(
                audio,
                sr,
                self.config["preprocessing"]["audio"]["sox_effects"],
            )
        return (audio, sr)

    def extract_spectral_features(self, audio_tensor: Tensor):
        """Given an audio tensor, extract the log Mel spectral features
        from the given start and end points

        Args:
            audio_tensor (Tensor): Tensor trimmed according
        """
        return self.spectral_transform(audio_tensor).squeeze()

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
        energy = linalg.norm(spectral_feature_tensor, dim=0)
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
            ["skipped processes", self.skipped_processes],
        ]
        return tabulate(table, headers, tablefmt=tablefmt)

    def preprocess(  # noqa: C901
        self,
        filelist=None,
        output_path="processed_filelist.psv",
        process_audio=False,
        process_sox_audio=False,
        process_spec=False,
        process_energy=False,
        process_f0=False,
        process_duration=False,
        process_pfs=False,
        process_text=False,
        overwrite=False,
    ):
        # TODO: use multiprocessing
        write_path = self.save_dir / output_path
        if write_path.exists() and not overwrite:
            logger.error(
                f"Preprocessed filelist at '{write_path}' already exists. Please either set overwrite=True or choose a new path"
            )
            exit()
        processed = 0
        files = []
        if filelist is None:
            filelist = self.filelist
        for f in tqdm(self.collect_files(filelist), total=len(filelist)):
            speaker = "default" if "speaker" not in f else f["speaker"]
            language = "default" if "language" not in f else f["language"]
            item = {**f, **{"speaker": speaker, "language": language}}
            audio = None
            spec = None
            if process_text:
                save_path = self.save_dir / self.sep.join(
                    [f["basename"], speaker, language, "text.npy"]
                )
                if overwrite or not save_path.exists():
                    torch.save(
                        self.extract_text_inputs(f["text"]),
                        save_path,
                    )
                    item["raw_text"] = f["text"]
                    item["clean_text"] = self.text_processor.clean_text(f["text"])
                else:
                    self.skipped_processes += 1
            if process_pfs:
                save_path = self.save_dir / self.sep.join(
                    [f["basename"], speaker, language, "pfs.npy"]
                )
                if overwrite or not save_path.exists():
                    torch.save(
                        self.extract_text_inputs(f["text"], use_pfs=True), save_path
                    )
                else:
                    self.skipped_processes += 1
            if process_sox_audio:
                save_path = self.save_dir / self.sep.join(
                    [f["basename"], speaker, language, "processed.wav"]
                )
                if overwrite or not save_path.exists():
                    save_audio(
                        save_path,
                        self.process_audio(
                            self.data_dir / (f["basename"] + ".wav"), use_effects=True
                        ),
                        self.config["preprocessing"]["audio"][
                            "alignment_sampling_rate"
                        ],
                        encoding="PCM_S",
                        bits_per_sample=self.config["preprocessing"]["audio"][
                            "alignment_bit_depth"
                        ],
                    )
                else:
                    self.skipped_processes += 1
            if process_audio:
                save_path = self.save_dir / self.sep.join(
                    [f["basename"], speaker, language, "audio.npy"]
                )
                if overwrite or not save_path.exists():
                    audio, _ = self.process_audio(
                        self.data_dir / (f["basename"] + ".wav")
                    )
                    torch.save(audio, save_path)
                else:
                    self.skipped_processes += 1
            if process_spec:
                save_path = self.save_dir / self.sep.join(
                    [
                        f["basename"],
                        speaker,
                        language,
                        f"spec-{self.audio_config['spec_type']}.npy",
                    ]
                )
                if overwrite or not save_path.exists():
                    if audio is None:
                        audio, _ = self.process_audio(
                            self.data_dir / (f["basename"] + ".wav")
                        )
                    spec = self.extract_spectral_features(audio)
                    torch.save(
                        spec,
                        save_path,
                    )
                else:
                    self.skipped_processes += 1
            if process_f0:
                save_path = self.save_dir / self.sep.join(
                    [f["basename"], speaker, language, "f0.npy"]
                )
                if overwrite or not save_path.exists():
                    if audio is None:
                        audio, _ = self.process_audio(
                            self.data_dir / (f["basename"] + ".wav")
                        )
                    torch.save(
                        self.extract_f0(audio),
                        save_path,
                    )
                else:
                    self.skipped_processes += 1
            if process_energy:
                save_path = self.save_dir / self.sep.join(
                    [f["basename"], speaker, language, "energy.npy"]
                )
                if overwrite or not save_path.exists():
                    if spec is None:
                        spec = self.extract_spectral_features(audio)
                    torch.save(self.extract_energy(spec), save_path),
                else:
                    self.skipped_processes += 1
            if process_duration:
                save_path = self.save_dir / self.sep.join(
                    [f["basename"], speaker, language, "duration.npy"]
                )
                dur_path = self.data_dir / f["basename"] + ".TextGrid"
                if not dur_path.exists():
                    logger.warning(f"File '{f}' if missing and will not be processed.")
                    self.missing_files.append(f)
                    continue
                if overwrite or not save_path.exists():
                    torch.save(self.extract_durations(dur_path), save_path)
                else:
                    self.skipped_processes += 1
            files.append(item)
            processed += 1
        logger.info(self.report(processed))
        write_filelist(files, self.save_dir / output_path)
        return files
