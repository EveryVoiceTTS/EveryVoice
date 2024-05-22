import json
import numbers
import tempfile
from enum import Enum
from pathlib import Path

import torch
from pytorch_lightning import Trainer

from everyvoice.config.type_definitions import DatasetTextRepresentation
from everyvoice.model.aligner.DeepForcedAligner.dfaligner.config import DFAlignerConfig
from everyvoice.model.aligner.DeepForcedAligner.dfaligner.model import Aligner
from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.config import (
    FastSpeech2Config,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
    FastSpeech2,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions_heavy import (
    Stats,
    StatsInfo,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import HiFiGANConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import HiFiGAN
from everyvoice.tests.basic_test_case import BasicTestCase
from everyvoice.tests.stubs import monkeypatch
from everyvoice.wizard import (
    ALIGNER_CONFIG_FILENAME_PREFIX,
    SPEC_TO_WAV_CONFIG_FILENAME_PREFIX,
    TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX,
)

monkey_saved_config = None


def find_non_basic_substructures(structure):
    """Return a list of all substructures within structure that are not basic data types.

    Dict, tuple, list, string, numbers are considered basic, stuff that depends on our
    code definitions are not.

    We use this function to make sure our checkpoints are robust to code refactorings.
    If we refactor and move the code for a class to a different file for performance
    or clarity reasons, but that class is embeded in a checkpoint, that checkpoint
    will no longer be loadable, so we want to block that early.
    """
    if isinstance(structure, Enum):
        return [structure]
    if isinstance(structure, (type(None), str, numbers.Number)):
        return []
    if isinstance(structure, (list, tuple)):
        result = []
        for element in structure:
            result.extend(find_non_basic_substructures(element))
        return result
    if isinstance(structure, dict):
        result = []
        for key, value in structure.items():
            result.extend(find_non_basic_substructures(key))
            result.extend(find_non_basic_substructures(value))
        return result
    return [structure]


class ModelTest(BasicTestCase):
    """Basic test for models"""

    def setUp(self) -> None:
        super().setUp()
        self.config = EveryVoiceConfig(
            contact=self.contact,
        )
        self.config_dir = self.data_dir / "relative" / "config"

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
                ),
                stats=Stats(
                    pitch=StatsInfo(
                        min=0, max=1, std=2, mean=3, norm_min=4, norm_max=5
                    ),
                    energy=StatsInfo(
                        min=7, max=8, std=9, mean=10, norm_min=11, norm_max=12
                    ),
                ),
                lang2id={"foo": 0, "bar": 1},
                speaker2id={"baz": 0, "qux": 1},
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

                def monkey_on_save_checkpoint(checkpoint):
                    ret = model.real_on_save_checkpoint(checkpoint)
                    global monkey_saved_config
                    monkey_saved_config = checkpoint["hyper_parameters"]["config"]
                    return ret

                with monkeypatch(
                    model, "real_on_save_checkpoint", model.on_save_checkpoint
                ):
                    with monkeypatch(
                        model, "on_save_checkpoint", monkey_on_save_checkpoint
                    ):
                        trainer.save_checkpoint(tmpdir / "model.ckpt")

                result = find_non_basic_substructures(monkey_saved_config)
                self.assertEqual(
                    result,
                    [],
                    f"The following parts of the checkpoint were not properly serialized by on_model_checkpoint(): {result}",
                )

                # We don't want just serializable, but actually serialized!
                ckpt = torch.load(tmpdir / "model.ckpt")
                try:
                    json.dumps(ckpt["hyper_parameters"])
                except (TypeError, OverflowError):
                    self.fail(
                        f"model {type(model).__name__} has some fields that are not JSON serializable"
                    )

    def test_find_non_basic_substructures(self):
        """Let's make sure the testing facility itself works..."""
        self.assertEqual(
            find_non_basic_substructures([{1: 2, "a": [3, 2]}, (1 / 3, 2.0), None]), []
        )
        si = StatsInfo(min=0, max=1, std=2, mean=3, norm_min=4, norm_max=5)
        result = find_non_basic_substructures(
            [
                {1: 2, "a": [3, 2]},
                (1 / 3, 2.0),
                Stats,
                {"a": si},
                {DatasetTextRepresentation.characters: "b"},
            ]
        )
        self.assertEqual(result, [Stats, si, DatasetTextRepresentation.characters])
