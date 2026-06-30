from pathlib import Path

from pytorch_lightning import Trainer

from everyvoice.config.shared_types import ContactInformation
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
    fp_trainer.save_checkpoint(dummy_fp_path)
    dummy_vocoder_path = tmp_dir / "vocoder.ckpt"
    vocoder_trainer.save_checkpoint(dummy_vocoder_path)
    torch.use_deterministic_algorithms(
        False
    )  # restore default, for test_synthesize on GPU

    return spec_model, dummy_fp_path, vocoder, dummy_vocoder_path


def get_stubbed_vocoder(tmp_dir: Path) -> tuple[HiFiGAN, Path]:
    """Creates and returns a stubbed HiFiGAN vocoder model and its save path for testing purposes.
    This function initializes a HiFiGAN vocoder with test contact information, sets up a barebones
    trainer, connects the model to the trainer's strategy, and saves the model checkpoint.
    Args:
        tmp_dir (Path): Temporary directory path where the vocoder checkpoint will be saved
    Returns:
        tuple[HiFiGAN, Path]: A tuple containing:
            - The initialized HiFiGAN vocoder model
            - Path to the saved vocoder checkpoint
    """

    contact_info = ContactInformation(
        contact_name="Test Runner", contact_email="info@everyvoice.ca"
    )
    vocoder = HiFiGAN(HiFiGANConfig(contact=contact_info))
    trainer = Trainer(default_root_dir=str(tmp_dir), barebones=True)
    trainer.strategy.connect(vocoder)
    vocoder_path = tmp_dir / "vocoder"
    trainer.save_checkpoint(vocoder_path)
    return vocoder, vocoder_path


def get_stubbed_model(tmp_dir: Path) -> tuple[FastSpeech2, Path]:

    contact_info = ContactInformation(
        contact_name="Test Runner", contact_email="info@everyvoice.ca"
    )

    model = FastSpeech2(
        config=FastSpeech2Config(contact=contact_info),
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
    trainer = Trainer(default_root_dir=str(tmp_dir), barebones=True)
    trainer.strategy.connect(model)
    model_path = tmp_dir / "model"
    trainer.save_checkpoint(model_path)
    return model, model_path
