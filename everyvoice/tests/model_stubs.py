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
from everyvoice.tests.stubs import silence_c_stderr


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
    with silence_c_stderr():
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
    with silence_c_stderr():
        trainer = Trainer(default_root_dir=str(tmp_dir), barebones=True)
    trainer.strategy.connect(model)
    model_path = tmp_dir / "model"
    trainer.save_checkpoint(model_path)
    return model, model_path


__all__ = ["get_stubbed_vocoder", "get_stubbed_model"]
