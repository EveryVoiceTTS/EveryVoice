"""Tests for the HiFiGAN vocoder (hfgl).

TO MIGRATE: these tests belong in the HiFiGAN_iSTFT_lightning submodule. They live
here only because that submodule doesn't have its own tests folder and CI yet.
Move this whole file there when it does; everyvoice.tests.grad_checks is generic
and can be imported from the submodule's tests, as the fs2 tests do.
"""

import pytest
import torch

from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl import api
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import HiFiGANConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import (
    HiFiGAN,
    HiFiGANGenerator,
)
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


@pytest.fixture()
def generator_only_checkpoint(stubbed_vocoder, tmp_path):
    """The stubbed vocoder's checkpoint converted for synthesis only, as by
    `everyvoice export spec-to-wav`: generator weights only."""
    _vocoder, vocoder_path = stubbed_vocoder
    checkpoint = torch.load(vocoder_path, weights_only=True)
    HiFiGAN.convert_ckpt_to_generator(checkpoint)
    path = tmp_path / "generator.ckpt"
    torch.save(checkpoint, path)
    return path


@pytest.mark.parametrize("checkpoint_type", ["full", "generator-only"])
def test_load_model(checkpoint_type, stubbed_vocoder, generator_only_checkpoint):
    stub, stub_path = stubbed_vocoder
    if checkpoint_type == "full":
        path, expected_class = stub_path, HiFiGAN
    else:
        path, expected_class = generator_only_checkpoint, HiFiGANGenerator

    model = api.load_model(str(path), accelerator="cpu")

    assert type(model) is expected_class
    assert model.config.model == stub.config.model
    assert model.config.preprocessing.audio == stub.config.preprocessing.audio
    assert not model.generator.training
    # remove_weight_norm() was applied
    assert not [
        name for name, _ in model.generator.named_parameters() if "weight_v" in name
    ]
    assert all(p.device == torch.device("cpu") for p in model.parameters())

    spec = torch.rand(1, model.config.preprocessing.audio.n_mels, 10)
    wavs, sample_rate = synthesize_data(spec, model, model.config)
    assert wavs.shape[0] == 1
    assert sample_rate == model.config.preprocessing.audio.output_sampling_rate


def test_load_model_missing_file(tmp_path):
    with pytest.raises(Exception):
        api.load_model(tmp_path / "does-not-exist.ckpt", accelerator="cpu")
