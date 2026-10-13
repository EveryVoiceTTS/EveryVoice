#!/usr/bin/env python

import json
import numbers
import sys
import tempfile
from enum import Enum
from pathlib import Path

import pytest
import torch

from everyvoice.config.type_definitions import DatasetTextRepresentation
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
from everyvoice.tests.stubs import (
    TEST_CONTACT,
    TEST_DATA_DIR,
    monkeypatch,
    mute_logger,
)
from everyvoice.wizard import (
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


def make_model(ModelType, config_dir: Path):
    """Instantiate a FastSpeech2 or HiFiGAN model from the test configs in config_dir"""
    with mute_logger("everyvoice.config.text_config"):
        if ModelType is FastSpeech2:
            # we should probably also test that the error about the variance adaptor is raised
            return FastSpeech2(
                FastSpeech2Config.load_config_from_path(
                    config_dir / f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml"
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
            )
        assert ModelType is HiFiGAN
        return HiFiGAN(
            HiFiGANConfig.load_config_from_path(
                config_dir / f"{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml"
            )
        )


class TestModel:
    """Basic test for models"""

    def setup_method(self) -> None:
        self.config = HiFiGANConfig(
            contact=TEST_CONTACT,
        )
        self.config_dir = TEST_DATA_DIR / "relative" / "config"

    def test_hparams(self):
        self.hifi_gan = HiFiGAN(self.config)
        assert self.config == self.hifi_gan.hparams.config
        assert self.config == self.hifi_gan.config

    def test_checkpoints_only_contain_serializable_content(self):
        """These tests help remove any dependencies on specific versions of Pydantic.
        By serializing our checkpoint hyperparameters and configuration with only JSON objects,
        we can help allow our models to be loaded by other versions of EveryVoice. This test ensures
        the hyperparameters only contain JSON serializable content
        """
        from pytorch_lightning import Trainer

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
                assert (
                    result == []
                ), f"The following parts of the checkpoint were not properly serialized by on_model_checkpoint(): {result}"

                # We don't want just serializable, but actually serialized!
                ckpt = torch.load(tmpdir / "model.ckpt", weights_only=True)
                try:
                    json.dumps(ckpt["hyper_parameters"])
                except (TypeError, OverflowError):
                    pytest.fail(
                        f"model {type(model).__name__} has some fields that are not JSON serializable"
                    )

    def test_find_non_basic_substructures(self):
        """Let's make sure the testing facility itself works..."""
        assert (
            find_non_basic_substructures([{1: 2, "a": [3, 2]}, (1 / 3, 2.0), None])
            == []
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
        assert result == [Stats, si, DatasetTextRepresentation.characters]


class TestLoadingModel:
    """Test loading models"""

    def setup_method(self) -> None:
        self.config_dir = TEST_DATA_DIR / "relative" / "config"

    def test_model_is_not_a_feature_prediction(self):
        """
        Loading a Vocoder Model instead of a FeaturePrediction Model.
        """
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir_str:
            model = HiFiGAN(
                HiFiGANConfig.load_config_from_path(
                    self.config_dir / f"{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml"
                )
            )
            trainer = Trainer(
                default_root_dir=tmpdir_str,
                enable_progress_bar=False,
                logger=False,
                max_epochs=1,
                limit_train_batches=1,
                limit_val_batches=1,
                callbacks=[ModelCheckpoint(dirpath=tmpdir_str, every_n_train_steps=1)],
            )
            trainer.strategy.connect(model)
            ckpt_fn = tmpdir_str + "/checkpoint.ckpt"
            trainer.save_checkpoint(ckpt_fn)
            import re

            with pytest.raises(
                TypeError,
                match=re.escape(
                    "Unable to load config.  Possible causes: is it really a FastSpeech2Config? or the correct version?"
                ),
            ):
                with mute_logger(
                    "everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model"
                ):
                    FastSpeech2.load_from_checkpoint(ckpt_fn)

    def test_model_is_not_a_vocoder(self):
        """
        Loading a FeaturePrediction Model instead of a Vocoder Model.
        """
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir_str:
            model = FastSpeech2(
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
            )
            trainer = Trainer(
                default_root_dir=tmpdir_str,
                enable_progress_bar=False,
                logger=False,
                max_epochs=1,
                limit_train_batches=1,
                limit_val_batches=1,
                callbacks=[ModelCheckpoint(dirpath=tmpdir_str, every_n_train_steps=1)],
            )
            trainer.strategy.connect(model)
            ckpt_fn = tmpdir_str + "/checkpoint.ckpt"
            trainer.save_checkpoint(ckpt_fn)
            import re

            with pytest.raises(
                TypeError,
                match=re.escape(
                    "Unable to load config.  Possible causes: is it really a VocoderConfig? or the correct version?"
                ),
            ):
                with mute_logger(
                    "everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model"
                ):
                    HiFiGAN.load_from_checkpoint(ckpt_fn)

    def test_wrong_model_type(self):
        """
        Detecting wrong model type in checkpoint.
        """
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir_str:
            if True:
                model = FastSpeech2(
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
                )
                trainer = Trainer(
                    default_root_dir=tmpdir_str,
                    enable_progress_bar=False,
                    logger=False,
                    max_epochs=1,
                    limit_train_batches=1,
                    limit_val_batches=1,
                    callbacks=[
                        ModelCheckpoint(dirpath=tmpdir_str, every_n_train_steps=1)
                    ],
                )
                trainer.strategy.connect(model)
                ckpt_fn = tmpdir_str + "/checkpoint.ckpt"
                trainer.save_checkpoint(ckpt_fn)
                m = torch.load(ckpt_fn, weights_only=True)
                assert "model_info" in m.keys()
                m["model_info"]["name"] = "BAD_TYPE"
                torch.save(m, ckpt_fn)
                m = torch.load(ckpt_fn, weights_only=True)
                assert "model_info" in m.keys()
                assert m["model_info"]["name"] == "BAD_TYPE"
                # assert m["model_info"]["version"] == "1.0"
                with pytest.raises(
                    TypeError,
                    match=r"Wrong model type \(BAD_TYPE\), we are expecting a 'FastSpeech2' model",
                ):
                    with mute_logger("everyvoice.config.text_config"):
                        FastSpeech2.load_from_checkpoint(ckpt_fn)

    @pytest.mark.parametrize(
        "ModelType", [FastSpeech2, HiFiGAN], ids=lambda t: t.__name__
    )
    def test_missing_model_version(self, ModelType):
        """
        Loading an old model that doesn't have a version.
        """
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint

        model = make_model(ModelType, self.config_dir)

        CANARY_VERSION = "CANARY_VERSION"
        with tempfile.TemporaryDirectory() as tmpdir_str:
            model._VERSION = CANARY_VERSION
            trainer = Trainer(
                default_root_dir=tmpdir_str,
                enable_progress_bar=False,
                logger=False,
                max_epochs=1,
                limit_train_batches=1,
                limit_val_batches=1,
                callbacks=[ModelCheckpoint(dirpath=tmpdir_str, every_n_train_steps=1)],
            )
            trainer.strategy.connect(model)
            ckpt_fn = tmpdir_str + "/checkpoint.ckpt"
            trainer.save_checkpoint(ckpt_fn)
            m = torch.load(ckpt_fn, weights_only=True)
            assert "model_info" in m.keys()
            assert m["model_info"]["name"] == ModelType.__name__
            assert m["model_info"]["version"] == CANARY_VERSION
            del m["model_info"]["version"]
            torch.save(m, ckpt_fn)
            if isinstance(model, FastSpeech2):
                # As of everyvoice==0.3.0 FastSpeech2 models are loaded by translating the embedding tables
                # but our mock here doesn't create a valid pre Version 1.2 FastSpeech2 checkpoint
                with mute_logger("everyvoice.config.text_config"):
                    with pytest.raises(AssertionError):
                        model = ModelType.load_from_checkpoint(ckpt_fn)
            else:
                with mute_logger("everyvoice.config.text_config"):
                    model = ModelType.load_from_checkpoint(ckpt_fn)
                assert model._VERSION, ["1.0" in "1.1"]

    @pytest.mark.parametrize(
        "ModelType", [FastSpeech2, HiFiGAN], ids=lambda t: t.__name__
    )
    def test_newer_model_version(self, ModelType):
        """
        Detecting an incompatible version number in the checkpoint.
        """
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint

        model = make_model(ModelType, self.config_dir)

        NEWER_VERSION = "100.0"
        with tempfile.TemporaryDirectory() as tmpdir_str:
            model._VERSION = NEWER_VERSION
            trainer = Trainer(
                default_root_dir=tmpdir_str,
                enable_progress_bar=False,
                logger=False,
                max_epochs=1,
                limit_train_batches=1,
                limit_val_batches=1,
                callbacks=[ModelCheckpoint(dirpath=tmpdir_str, every_n_train_steps=1)],
            )
            trainer.strategy.connect(model)
            ckpt_fn = tmpdir_str + "/checkpoint.ckpt"
            trainer.save_checkpoint(ckpt_fn)
            m = torch.load(ckpt_fn, weights_only=True)
            assert "model_info" in m.keys()
            assert m["model_info"]["name"] == ModelType.__name__
            assert m["model_info"]["version"] == NEWER_VERSION
            with pytest.raises(
                ValueError,
                match=r"Your model was created with a newer version of EveryVoice, please update your software.",
            ):
                with mute_logger("everyvoice.config.text_config"):
                    ModelType.load_from_checkpoint(ckpt_fn)


CONFIGS_TO_TEST = pytest.mark.parametrize(
    "ConfigType, filename",
    [
        (FastSpeech2Config, TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX),
        (HiFiGANConfig, SPEC_TO_WAV_CONFIG_FILENAME_PREFIX),
    ],
    ids=["FastSpeech2Config", "HiFiGANConfig"],
)


class TestLoadingConfig:
    """Test loading configurations"""

    def setup_method(self) -> None:
        self.config_dir = TEST_DATA_DIR / "relative" / "config"

    @CONFIGS_TO_TEST
    def test_config_versionless(self, ConfigType, filename):
        """
        Validate that we can load a config that doesn't have a `VERSION` as a version 1.0 config.
        """

        arguments = ConfigType.load_config_from_path(
            self.config_dir / f"{filename}.yaml"
        ).model_dump()
        del arguments["VERSION"]

        assert "VERSION" not in arguments
        c = ConfigType(**arguments)
        assert c.VERSION == "1.0"

    @CONFIGS_TO_TEST
    def test_config_newer_version(self, ConfigType, filename):
        """
        Validate that we are detecting that a config is newer.
        """

        reference = ConfigType.load_config_from_path(
            self.config_dir / f"{filename}.yaml"
        )
        NEWER_VERSION = "100.0"
        reference.VERSION = NEWER_VERSION

        with pytest.raises(
            ValueError,
            match=r"Your config was created with a newer version of EveryVoice, please update your software.",
        ):
            ConfigType(**reference.model_dump())


class TestVersion:
    def test_newer_version(self):
        """
        Canary test.  We use packaging.version.Version to compare model's versions and config's versions.
        """
        from packaging.version import Version

        assert not "10.0" > "9.0"
        assert Version("10.0") > Version("9.0")


if __name__ == "__main__":
    pytest.main(sys.argv)
