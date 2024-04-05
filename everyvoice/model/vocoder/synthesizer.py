from typing import Tuple

import numpy as np
import torch
from loguru import logger

from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.config import (
    FastSpeech2Config,
)


# TODO: This should go under everyvoice/model/vocoder/.
#       It would define a common interface for all vocoder.
#       Each specific vocoder should implement SynthesizerBase.
class SynthesizerBase:
    """
    A common interface between the generator_universal and Everyvoice's vocoder.
    """

    def __call__(self, inputs: torch.Tensor) -> Tuple[np.ndarray, int]:
        raise NotImplementedError


# TODO: This should be implemented/moved under
#       everyvoice/model/vocoder/original_hifigan_helper/
class SynthesizerUniversal(SynthesizerBase):
    """
    A synthesizer that uses the generator_universal.
    """

    def __init__(self, vocoder_ckpt, config, device: torch.device) -> None:
        from everyvoice.model.vocoder.original_hifigan_helper import (
            UNIVERSAL_CONFIG,
            AttrDict,
            Generator,
        )

        self.vocoder = Generator(AttrDict(UNIVERSAL_CONFIG))
        self.vocoder.load_state_dict(vocoder_ckpt["generator"])
        self.vocoder.eval()
        self.vocoder.remove_weight_norm()
        self.vocoder.to(device)

        # TODO: If we don't need all of config but simply output_sampling_rate,
        # may be we should only store that.
        self.sampling_rate = config.preprocessing.audio.output_sampling_rate

    def __call__(self, inputs: torch.Tensor) -> Tuple[np.ndarray, int]:
        """
        Generate wavs using the generator_universal model.
        """
        # mels (1, 80, 111) normal
        # mels small (1, 80, 5)
        with torch.no_grad():
            wavs = self.vocoder(inputs.transpose(1, 2)).squeeze(1)
        wavs = wavs.cpu().numpy()  # B, T

        return wavs, self.sampling_rate


# TODO: We should have a less generic name for the EveryVoice synthesizer.
# TODO: This should be implemented under
#       everyvoice/model/vocoder/HiFiGAN_iSTFT_lightning/hfgl/ as it is
#       specific to hfgl.
class Synthesizer(SynthesizerBase):
    """
    A synthesizer that uses EveryVoice models.
    """

    def __init__(self, vocoder_ckpt, device: torch.device) -> None:
        from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import HiFiGAN

        self.config = vocoder_ckpt["hyper_parameters"]["config"]
        self.model = HiFiGAN(self.config).to(device)
        self.model.load_state_dict(vocoder_ckpt["state_dict"])
        self.model.generator.eval()
        self.model.generator.remove_weight_norm()

    def __call__(self, inputs: torch.Tensor) -> Tuple[np.ndarray, int]:
        """Synthesize a batch of waveforms from spectral features

        Args:
            inputs (Tensor): data tensor, expects output from feature prediction network to be size (b=batch_size, t=number_of_frames, k=n_mels)
        Returns:
            Tuple[np.ndarray, int]: a 1-D array of the wav file and the sampling rate
        """

        from everyvoice.utils.heavy import get_spectral_transform

        if self.config.model.istft_layer:
            inverse_spectral_transform = get_spectral_transform(
                "istft",
                self.model.generator.post_n_fft,
                self.model.generator.post_n_fft,
                self.model.generator.post_n_fft // 4,
                # NOTE: Should it be the inputs or model's device?
            ).to(inputs.device)
            with torch.no_grad():
                mag, phase = self.model.generator(inputs.transpose(1, 2))
            wav = inverse_spectral_transform(mag * torch.exp(phase * 1j)).unsqueeze(-2)
        else:
            with torch.no_grad():
                wav = self.model.generator(inputs.transpose(1, 2))

        return (
            wav.squeeze().cpu().numpy(),
            self.config.preprocessing.audio.output_sampling_rate,
        )


def get_synthesizer(
    config: FastSpeech2Config,
    device: torch.device,
) -> SynthesizerBase:
    if config.training.vocoder_path is None:
        import sys

        # TODO: Should we replace this by an assertion instead?
        logger.error(
            "No vocoder was provided, please specify "
            "--vocoder-path /path/to/vocoder on the command line."
        )
        sys.exit(1)
    else:
        vocoder_ckpt = torch.load(config.training.vocoder_path, map_location=device)
        if "generator" in vocoder_ckpt.keys():
            # Necessary when passing --filelist
            return SynthesizerUniversal(vocoder_ckpt, config, device)

        return Synthesizer(vocoder_ckpt, device)
