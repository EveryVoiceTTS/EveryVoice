#!/usr/bin/env python

import json
import tempfile
from pathlib import Path

import torch
from pytorch_lightning import Trainer

from everyvoice.model.aligner.DeepForcedAligner.dfaligner.config import DFAlignerConfig
from everyvoice.model.aligner.DeepForcedAligner.dfaligner.model import Aligner
from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.config import (
    FastSpeech2Config,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
    FastSpeech2,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import HiFiGANConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import HiFiGAN
from everyvoice.tests.basic_test_case import BasicTestCase
from everyvoice.wizard import (
    ALIGNER_CONFIG_FILENAME_PREFIX,
    SPEC_TO_WAV_CONFIG_FILENAME_PREFIX,
    TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX,
)


class ModelTest(BasicTestCase):
    """Basic test for models"""

    def setUp(self) -> None:
        super().setUp()
        self.config = EveryVoiceConfig(
            contact=self.contact,
        )
        self.config_dir = Path(__file__).parent / "data" / "relative" / "config"

    def test_hparams(self):
        self.hifi_gan = HiFiGAN(self.config.vocoder)
        self.assertEqual(self.config.vocoder, self.hifi_gan.hparams.config)
        self.assertEqual(self.config.vocoder, self.hifi_gan.config)

    def test_checkpoints_only_contain_serializable_content(self):
        """These tests help remove any dependencies on specific versions of Pydantic.
        By serializing our checkpoint hyperparameters and configuration with only JSON objects,
        we can help allow our models to be loaded by other versions of EveryVoice. This test ensures
        the hyperparameters only contain JSON serializable content
        """
        SERIAL_SAFE_MODELS = [
            HiFiGAN(
                HiFiGANConfig.load_config_from_path(
                    self.config_dir / f"{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml"
                )
            ),
            FastSpeech2(
                FastSpeech2Config.load_config_from_path(
                    self.config_dir / f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml"
                )
            ),  # we should probably also test that the error about the variance adaptor is raised
            Aligner(
                DFAlignerConfig.load_config_from_path(
                    self.config_dir / f"{ALIGNER_CONFIG_FILENAME_PREFIX}.yaml"
                )
            ),
        ]
        for model in SERIAL_SAFE_MODELS:
            trainer = Trainer()
            with tempfile.TemporaryDirectory() as tmpdir_str:
                # Hacky way to connect the trainer with a model instead of trainer.fit(model) just for testing
                # https://lightning.ai/forums/t/saving-a-lightningmodule-without-a-trainer/2217/2
                trainer.strategy.connect(model)
                tmpdir = Path(tmpdir_str)
                trainer.save_checkpoint(tmpdir / "model.ckpt")
                ckpt = torch.load(tmpdir / "model.ckpt")
                try:
                    json.dumps(ckpt["hyper_parameters"])
                except (TypeError, OverflowError):
                    self.fail(
                        f"model {type(model).__name__} has some fields that are not JSON serializable"
                    )
