""" Preprocessor Module that given a filelist containing text, wav, textgrid:
    - extracts log Mel spectral features
    - extracts f0 (phone-level or frame-level)
    - extracts durations
    - extracts energy
    - extracts inputs (ex. phonological feats)
"""

from typing import Tuple

import numpy as np
import pyworld as pw
import torchaudio.functional as F
import torchaudio.transforms as T
from torch import Tensor, linalg, mean, tensor
from torchaudio import load as load_audio
from torchaudio.sox_effects import apply_effects_tensor

from config import ConfigError
from utils import read_textgrid


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.audio_config = config["preprocessing"]["audio"]
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

    def extract_text_inputs(self, text):
        """Given some text, normalize it, g2p it, and save as one-hot or multi-hot phonological feature vectors

        Args:
            text (str): text
        """
        pass
