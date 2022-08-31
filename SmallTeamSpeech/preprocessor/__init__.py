""" Preprocessor Module that given a filelist containing text, wav, textgrid:
    - extracts log Mel spectral features
    - extracts f0 (phone-level or continuous wavelet)
    - extracts durations
    - extracts energy
    - extracts inputs (ex. phonological feats)
"""

from typing import Tuple

from loguru import logger
from torch import Tensor, linalg
from torchaudio import load as load_audio
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import MelSpectrogram, Spectrogram


class Preprocessor:
    def __init__(self, config):
        self.config = config
        audio_config = config["preprocessing"]["audio"]
        # Define Spectral Transform
        if audio_config["spec_type"] == "mel":
            self.spectral_transform = MelSpectrogram(
                sample_rate=audio_config["target_sampling_rate"],
                n_fft=audio_config["n_fft"],
                win_length=audio_config["fft_window_frames"],
                hop_length=audio_config["fft_hop_frames"],
                n_mels=audio_config["n_mels"],
            )
        elif config["preprocessing"]["audio"]["spec_type"] == "linear":
            self.spectral_transform = Spectrogram(
                n_fft=audio_config["n_fft"],
                win_length=audio_config["fft_window_frames"],
                hop_length=audio_config["fft_hop_frames"],
            )
        elif config["preprocessing"]["audio"]["spec_type"] == "raw":
            self.spectral_transform = Spectrogram(
                n_fft=audio_config["n_fft"],
                win_length=audio_config["fft_window_frames"],
                hop_length=audio_config["fft_hop_frames"],
                power=None,
            )
        else:
            logger.error(
                f"Spectral feature specification {config['preprocessing']['audio']['spec_type'] == 'mel'} is not supported"
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

    def extract_phone_f0(self, audio_tensor: Tensor, durations):
        """Given an audio tensor, and durations extract the f0 averaged across a phone

        Args:
            spectral_feature_tensor (Tensor): tensor of spectral features extracted from audio
            durations (_type_): _descriptiont    #TODO
        """
        pass

    def extract_durations(self, textgrid_path: str):
        """Extract durations from a textgrid path

        Args:
            textgrid_path (str): path to a textgrid file
        """
        pass

    def extract_energy(self, spectral_feature_tensor: Tensor, durations):
        """Given a spectral feature tensor, and durations extract the energy averaged across a phone

        Args:
            spectral_feature_tensor (Tensor): tensor of spectral features extracted from audio
            durations (_type_): _descriptiont    #TODO
        """
        energy = linalg.norm(spectral_feature_tensor, dim=1)
        if durations:
            pass  # average them based on phones
        return energy

    def extract_text_inputs(self, text):
        """Given some text, normalize it, g2p it, and save as one-hot or multi-hot phonological feature vectors

        Args:
            text (str): text
        """
        pass
