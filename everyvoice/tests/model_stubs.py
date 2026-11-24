from pathlib import Path

from pytorch_lightning import Trainer

from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
    FastSpeech2,
    FastSpeech2Config,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions_heavy import (
    Stats,
    StatsInfo,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import HiFiGANConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import HiFiGAN
from everyvoice.wizard import (
    SPEC_TO_WAV_CONFIG_FILENAME_PREFIX,
    TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX,
)

from .stubs import CONFIG_DIR

# The global_step written into the stubbed checkpoints. A fresh Trainer saves step
# 0, which is indistinguishable from an unset step counter; distinct values per
# model also catch mixing up the text-to-spec model and the vocoder.
STUBBED_MODEL_GLOBAL_STEP = 987_654
STUBBED_VOCODER_GLOBAL_STEP = 876_543


def save_stub_checkpoint(trainer: Trainer, path: Path, global_step: int) -> None:
    """Save the model connected to trainer at path, as if trained for global_step
    steps: Lightning has no supported way to set a Trainer's step without
    training, so it is set in the saved checkpoint."""
    import torch

    trainer.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["global_step"] = global_step
    torch.save(checkpoint, path)


def get_dummy_models(tmp_dir: Path) -> tuple[FastSpeech2, Path, HiFiGAN, Path]:
    """Usage: dummy_fp, dummy_fp_path, dummy_vocoder, dummy_vocoder_path = get_dummy_models(tmp_dir)"""
    import random

    import torch

    # Set a manual seed, because some seeds cause the model
    # to fail to generate a proper wav file. This seed was taken
    # from running torch.seed() on a working run. Note: this is a bit
    # brittle, but this test is just to test that the synthesize command
    # works given two functional checkpoints. Further tests into the effects
    # of seeds should be looked into.
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(10719787423044995460)
    random.seed(10719787423044995460)
    vocoder = HiFiGAN(
        HiFiGANConfig.load_config_from_path(
            CONFIG_DIR / f"{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml"
        )
    )
    spec_model = FastSpeech2(
        FastSpeech2Config.load_config_from_path(
            CONFIG_DIR / f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml"
        ),
        lang2id={"default": 0},
        speaker2id={"default": 0},
        stats=Stats(
            pitch=StatsInfo(
                min=150, max=300, std=2.0, mean=0.5, norm_max=1.0, norm_min=0.1
            ),
            energy=StatsInfo(
                min=0.1, max=10.0, std=2.0, mean=0.5, norm_max=1.0, norm_min=0.1
            ),
        ),
    )
    tmp_dir_str = str(tmp_dir)
    vocoder_trainer = Trainer(default_root_dir=tmp_dir_str, barebones=True)
    fp_trainer = Trainer(default_root_dir=tmp_dir_str, barebones=True)
    vocoder_trainer.strategy.connect(vocoder)
    fp_trainer.strategy.connect(spec_model)
    dummy_fp_path = tmp_dir / "fp.ckpt"
    save_stub_checkpoint(fp_trainer, dummy_fp_path, STUBBED_MODEL_GLOBAL_STEP)
    dummy_vocoder_path = tmp_dir / "vocoder.ckpt"
    save_stub_checkpoint(
        vocoder_trainer, dummy_vocoder_path, STUBBED_VOCODER_GLOBAL_STEP
    )
    torch.use_deterministic_algorithms(
        False
    )  # restore default, for test_synthesize on GPU

    return spec_model, dummy_fp_path, vocoder, dummy_vocoder_path
