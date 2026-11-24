"""Tests for the HiFiGAN vocoder (hfgl).

TO MIGRATE: these tests belong in the HiFiGAN_iSTFT_lightning submodule. They live
here only because that submodule doesn't have its own tests folder and CI yet.
Move this whole file there when it does; everyvoice.tests.grad_checks is generic
and can be imported from the submodule's tests, as the fs2 tests do.
"""

import pytest
import torch
from pytorch_lightning import Trainer

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
from everyvoice.tests.model_stubs import STUBBED_VOCODER_GLOBAL_STEP
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
def vocoder_checkpoints(stubbed_vocoder, tmp_path):
    """The stubbed vocoder's checkpoint, in full and converted for synthesis only
    (generator weights only), as by `everyvoice export spec-to-wav`."""
    _vocoder, vocoder_path = stubbed_vocoder
    checkpoint = torch.load(vocoder_path, weights_only=True)
    HiFiGAN.convert_ckpt_to_generator(checkpoint)
    generator_path = tmp_path / "generator.ckpt"
    torch.save(checkpoint, generator_path)
    return {"full": vocoder_path, "generator-only": generator_path}


@pytest.mark.parametrize(
    "checkpoint_type, expected_class",
    [("full", HiFiGAN), ("generator-only", HiFiGANGenerator)],
)
def test_load_model(
    checkpoint_type, expected_class, stubbed_vocoder, vocoder_checkpoints
):
    stub, _stub_path = stubbed_vocoder
    model = api.load_model(str(vocoder_checkpoints[checkpoint_type]), accelerator="cpu")

    assert type(model) is expected_class
    assert model.config.model == stub.config.model
    assert model.config.preprocessing.audio == stub.config.preprocessing.audio
    assert model.global_step_at_load == STUBBED_VOCODER_GLOBAL_STEP
    assert model.global_step == 0  # no Trainer attached
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


class TestGlobalStepAtLoad:
    def test_fresh_model(self):
        assert HiFiGAN(HiFiGANConfig(contact=TEST_CONTACT)).global_step_at_load is None

    def test_load_from_checkpoint(self, vocoder_checkpoints):
        """The path used to fine-tune a vocoder"""
        model = HiFiGAN.load_from_checkpoint(vocoder_checkpoints["full"])
        assert model.global_step_at_load == STUBBED_VOCODER_GLOBAL_STEP

    def test_not_saved_in_checkpoints(self, vocoder_checkpoints, tmp_path):
        model = HiFiGAN.load_from_checkpoint(vocoder_checkpoints["full"])
        trainer = Trainer(default_root_dir=str(tmp_path), barebones=True)
        trainer.strategy.connect(model)
        trainer.save_checkpoint(tmp_path / "resaved.ckpt")

        checkpoint = torch.load(tmp_path / "resaved.ckpt", weights_only=True)
        assert checkpoint["global_step"] == 0  # the new Trainer's step
        assert list(checkpoint["hyper_parameters"]) == ["config"]
        assert str(STUBBED_VOCODER_GLOBAL_STEP) not in repr(
            checkpoint["hyper_parameters"]
        )


def test_load_model_missing_file(tmp_path):
    with pytest.raises(Exception):
        api.load_model(tmp_path / "does-not-exist.ckpt", accelerator="cpu")
