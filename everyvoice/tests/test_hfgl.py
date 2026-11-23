"""Tests for the HiFiGAN vocoder (hfgl).

TO MIGRATE: these tests belong in the HiFiGAN_iSTFT_lightning submodule. They live
here only because that submodule doesn't have its own tests folder and CI yet.
Move this whole file there when it does; everyvoice.tests.grad_checks is generic
and can be imported from the submodule's tests, as the fs2 tests do.
"""

import pytest
import torch

from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import HiFiGANConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import HiFiGAN
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
    synthesize_data,
)
from everyvoice.tests.grad_checks import assert_no_grad_tracking, assert_no_param_grads
from everyvoice.tests.stubs import TEST_CONTACT


@pytest.mark.parametrize(
    "model_config",
    [
        pytest.param({}, id="waveform"),
        # istft_layer upsamples by 4 itself, so it needs fewer upsampling layers
        pytest.param(
            {
                "istft_layer": True,
                "upsample_rates": [8, 8],
                "upsample_kernel_sizes": [16, 16],
            },
            id="istft",
        ),
    ],
)
def test_synthesize_data_does_not_track_gradients(model_config):
    config = HiFiGANConfig(contact=TEST_CONTACT, model=model_config)
    vocoder = HiFiGAN(config)
    # As in load_hifigan_from_checkpoint()
    vocoder.generator.eval()
    vocoder.generator.remove_weight_norm()
    assert any(param.requires_grad for param in vocoder.parameters())

    batch_size, n_frames = 2, 10
    spec = torch.rand(batch_size, config.preprocessing.audio.n_mels, n_frames)
    wavs, _sample_rate = synthesize_data(spec, vocoder, config)

    assert torch.is_grad_enabled()
    assert wavs.shape[0] == batch_size
    assert_no_grad_tracking(wavs, "wavs")
    assert_no_param_grads(vocoder, "vocoder")
