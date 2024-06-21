import json
import tempfile
import time
from pathlib import Path
from typing import Callable
from unittest import TestCase

import yaml
from pydantic import ValidationError

from everyvoice import exceptions
from everyvoice.config.preprocessing_config import (
    AudioConfig,
    Dataset,
    PreprocessingConfig,
)
from everyvoice.config.shared_types import (
    BaseTrainingConfig,
    LoggerConfig,
    init_context,
)
from everyvoice.config.text_config import TextConfig
from everyvoice.model.aligner.config import AlignerConfig
from everyvoice.model.aligner.DeepForcedAligner.dfaligner.config import (
    DFAlignerTrainingConfig,
)
from everyvoice.model.e2e.config import E2ETrainingConfig, EveryVoiceConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import (
    HiFiGANModelConfig,
    HiFiGANTrainingConfig,
)
from everyvoice.tests.basic_test_case import BasicTestCase
from everyvoice.utils import (
    expand_config_string_syntax,
    load_config_from_json_or_yaml_path,
    lower,
    nfc_normalize,
)
from everyvoice.wizard import (
    ALIGNER_CONFIG_FILENAME_PREFIX,
    PREPROCESSING_CONFIG_FILENAME_PREFIX,
    SPEC_TO_WAV_CONFIG_FILENAME_PREFIX,
    TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX,
    TEXT_TO_WAV_CONFIG_FILENAME_PREFIX,
)


def _writer_helper(model, filename):
    with open(filename, "w", encoding="utf8") as f:
        f.write(model.model_dump_json())


class ConfigTest(BasicTestCase):
    """Basic test for hyperparameter configuration"""

    def setUp(self) -> None:
        super().setUp()
        self.config = EveryVoiceConfig(
            contact=self.contact,
            aligner=AlignerConfig(contact=self.contact),
            feature_prediction=FeaturePredictionConfig(contact=self.contact),
            vocoder=VocoderConfig(contact=self.contact),
        )

    def test_from_object(self):
        """Test from object"""
        config_default = EveryVoiceConfig(
            contact=self.contact,
            aligner=AlignerConfig(contact=self.contact),
            feature_prediction=FeaturePredictionConfig(contact=self.contact),
            vocoder=VocoderConfig(contact=self.contact),
        )
        config_declared = EveryVoiceConfig(
            contact=self.contact,
            aligner=AlignerConfig(contact=self.contact),
            feature_prediction=FeaturePredictionConfig(contact=self.contact),
            vocoder=VocoderConfig(contact=self.contact),
            training=E2ETrainingConfig(),
        )
        config_32 = EveryVoiceConfig(
            contact=self.contact,
            aligner=AlignerConfig(contact=self.contact),
            feature_prediction=FeaturePredictionConfig(contact=self.contact),
            vocoder=VocoderConfig(contact=self.contact),
            training=E2ETrainingConfig(batch_size=32),
        )
        self.assertEqual(config_default.training.batch_size, 16)
        self.assertEqual(config_declared.training.batch_size, 16)
        self.assertEqual(config_32.training.batch_size, 32)

    def test_config_save_dirs(self):
        with tempfile.TemporaryDirectory(prefix="test_config_save_dirs") as tempdir:
            # Preprocessing Config
            tempdir = Path(tempdir)
            with self.assertRaises(ValidationError):
                preprocessing_config = PreprocessingConfig(save_dir=1)
            with init_context({"writing_config": tempdir}):
                preprocessing_config = PreprocessingConfig(save_dir="./bloop")
            self.assertTrue((tempdir / preprocessing_config.save_dir).exists())

    def test_config_partial(self):
        with tempfile.TemporaryDirectory(
            prefix="test_config_partial"
        ) as tempdir_str, init_context({"writing_config": Path(tempdir_str)}):
            # Preprocessing Config
            tempdir = Path(tempdir_str)
            _writer_helper(AudioConfig(), tempdir / "audio.json")
            config = PreprocessingConfig(
                path_to_audio_config_file=(tempdir / "audio.json")
            )
            self.assertTrue(isinstance(config.audio, AudioConfig))
            # bad partial
            with self.assertRaises(exceptions.InvalidConfiguration):
                with tempfile.NamedTemporaryFile(
                    prefix="test", mode="w", suffix=".yaml"
                ) as tf:
                    tf.write(" ")
                    tf.flush()
                    PreprocessingConfig(path_to_audio_config_file=tf.name)
            # Write shared:
            _writer_helper(PreprocessingConfig(), tempdir / "preprocessing.json")
            _writer_helper(TextConfig(), tempdir / "text.json")
            _writer_helper(BaseTrainingConfig(), tempdir / "training.json")
            # Aligner Config
            _writer_helper(
                AlignerConfig(contact=self.contact).training,
                tempdir / "aligner-training.json",
            )
            _writer_helper(
                AlignerConfig(contact=self.contact).model,
                tempdir / "aligner-model.json",
            )
            aligner_config = AlignerConfig(
                contact=self.contact,
                path_to_model_config_file=(tempdir / "aligner-model.json"),
                path_to_preprocessing_config_file=(tempdir / "preprocessing.json"),
                path_to_text_config_file=(tempdir / "text.json"),
                path_to_training_config_file=tempdir / "aligner-training.json",
            )
            _writer_helper(aligner_config, tempdir / "aligner.json")
            self.assertTrue(isinstance(aligner_config, AlignerConfig))
            # FP Config
            _writer_helper(
                FeaturePredictionConfig(contact=self.contact).training,
                tempdir / "fp-training.json",
            )
            _writer_helper(
                FeaturePredictionConfig(contact=self.contact).model,
                tempdir / "fp-model.json",
            )
            fp_config = FeaturePredictionConfig(
                contact=self.contact,
                path_to_model_config_file=(tempdir / "fp-model.json"),
                path_to_preprocessing_config_file=(tempdir / "preprocessing.json"),
                path_to_text_config_file=(tempdir / "text.json"),
                path_to_training_config_file=tempdir / "fp-training.json",
            )
            _writer_helper(fp_config, tempdir / "fp.json")
            self.assertTrue(isinstance(fp_config, FeaturePredictionConfig))
            # Vocoder Config
            _writer_helper(
                VocoderConfig(contact=self.contact).training,
                tempdir / "vocoder-training.json",
            )
            _writer_helper(
                VocoderConfig(contact=self.contact).model,
                tempdir / "vocoder-model.json",
            )
            vocoder_config = VocoderConfig(
                contact=self.contact,
                path_to_model_config_file=(tempdir / "vocoder-model.json"),
                path_to_preprocessing_config_file=(tempdir / "preprocessing.json"),
                path_to_training_config_file=tempdir / "vocoder-training.json",
            )
            _writer_helper(vocoder_config, tempdir / "vocoder.json")
            self.assertTrue(isinstance(vocoder_config, VocoderConfig))
            # E2E Config
            e2e_config = EveryVoiceConfig(
                contact=self.contact,
                path_to_aligner_config_file=(tempdir / "aligner.json"),
                path_to_feature_prediction_config_file=(tempdir / "fp.json"),
                path_to_training_config_file=(tempdir / "training.json"),
                path_to_vocoder_config_file=(tempdir / "vocoder.json"),
            )
            self.assertTrue(isinstance(e2e_config, EveryVoiceConfig))

    def test_config_partial_override(self):
        """Test override of partial"""
        with tempfile.NamedTemporaryFile(
            prefix="test_config_partial_override", mode="w", suffix=".yaml"
        ) as tf:
            tf.write(AudioConfig().model_dump_json())
            tf.flush()
            # override with actual class
            config = PreprocessingConfig(
                path_to_audio_config_file=tf.name,
                audio=AudioConfig(min_audio_length=1.0),
            )
            self.assertEqual(config.audio.min_audio_length, 1.0)
            # override with dict
            config = PreprocessingConfig(
                path_to_audio_config_file=tf.name, audio={"max_audio_length": 1.0}
            )
            self.assertEqual(config.audio.max_audio_length, 1.0)
            # pass something invalid
            with self.assertRaises(ValidationError):
                PreprocessingConfig(path_to_audio_config_file=tf.name, audio=1.0)

    def test_update_from_file(self):
        """Test that updating the config from yaml/json works"""
        with open(self.data_dir / "update.json") as f:
            update = json.load(f)
        self.config.update_config(update)
        with open(self.data_dir / "update.yaml") as f:
            update = yaml.safe_load(f)
        self.config.update_config(update)
        self.assertEqual(self.config.feature_prediction.training.batch_size, 123)
        self.assertEqual(self.config.vocoder.training.batch_size, 456)

    def test_string_to_callable(self):
        # Test Basic Functionality
        config = FeaturePredictionConfig(
            contact=self.contact, text=TextConfig(cleaners=["everyvoice.utils.lower"])
        )
        self.assertEqual(config.text.cleaners, [lower])
        # Test missing function
        with self.assertRaises(AttributeError):
            config.update_config({"text": {"cleaners": ["everyvoice.utils.foobarfoo"]}})
        # Test missing module
        with self.assertRaises(ImportError):
            config.update_config({"text": {"cleaners": ["foobarfoo.utils.lower"]}})
        # Test not string
        with self.assertRaises(ValueError):
            config.update_config({"text": {"cleaners": [1]}})
        # Test plain string
        config = LoggerConfig(sub_dir_callable="foobar")
        self.assertEqual(config.sub_dir_callable(), "foobar")

    def test_call_sub_dir(self):
        config = LoggerConfig()
        # sub_dir should get called from sub_dir_callable and be a string of an int
        self.assertTrue(isinstance(int(config.sub_dir), int))
        # Just in case we're super speedy
        time.sleep(1)
        self.assertGreater(int(config.sub_dir_callable()), int(config.sub_dir))
        serialized_config = config.model_dump()
        # Exclude sub_dir by default when serializing as it should get overriden on each run
        self.assertTrue("sub_dir" not in serialized_config)

    def test_properly_deserialized_callables(self):
        config = TextConfig(cleaners=[nfc_normalize, "everyvoice.utils.lower"])
        for fn in config.cleaners:
            self.assertTrue(isinstance(fn, Callable))

    def test_string_to_dict(self):
        base_config = EveryVoiceConfig(
            contact=self.contact,
            aligner=AlignerConfig(contact=self.contact),
            feature_prediction=FeaturePredictionConfig(contact=self.contact),
            vocoder=VocoderConfig(contact=self.contact),
        )
        test_string = "vocoder.training.gan_type=wgan"
        test_bad_strings = [
            "vocoder.training.gan_type==wgan",
            "vocoder.training.gan_typewgan",
        ]
        # test_missing = ["training.foobar.gan_type=original"]
        test_dict = expand_config_string_syntax(test_string)
        self.assertEqual(test_dict, {"vocoder": {"training": {"gan_type": "wgan"}}})
        for bs in test_bad_strings:
            with self.assertRaises(ValueError):
                expand_config_string_syntax(bs)

        self.assertEqual(base_config.vocoder.training.gan_type.value, "original")
        config = base_config.combine_configs(base_config, test_dict)
        self.assertEqual(config["vocoder"]["training"]["gan_type"], "wgan")

    def test_changes(self):
        """Test that the changes to the config are correct"""
        self.config.update_config(
            {"feature_prediction": {"text": {"cleaners": ["everyvoice.utils.lower"]}}}
        )
        self.assertEqual(self.config.feature_prediction.text.cleaners, [lower])

    def test_load_empty_config(self):
        with tempfile.NamedTemporaryFile(
            prefix="test_load_empty_config", mode="w", suffix=".yaml"
        ) as tf:
            tf.write(" ")
            tf.flush()
            with self.assertRaises(exceptions.InvalidConfiguration):
                load_config_from_json_or_yaml_path(Path(tf.name))

    def test_change_with_indices(self):
        """Text the --config-args can also work with arrays"""
        with tempfile.TemporaryDirectory(
            prefix="test_change_with_indices"
        ) as tempdir, init_context({"writing_config": Path(tempdir)}):
            config = FeaturePredictionConfig(contact=self.contact)
            config.update_config(
                {
                    "preprocessing": {
                        "source_data": {"0": {"filelist": "/foo/bar/filelist.psv"}}
                    }
                }
            )
            self.assertEqual(
                str(config.preprocessing.source_data[0].filelist),
                "/foo/bar/filelist.psv",
            )

    def test_shared_sox(self) -> None:
        """Test that the shared sox config is correct"""
        vocoder_config = VocoderConfig(
            contact=self.contact,
            preprocessing=PreprocessingConfig(
                source_data=[
                    Dataset(permissions_obtained=True),
                    Dataset(permissions_obtained=True),
                    Dataset(permissions_obtained=True),
                    Dataset(permissions_obtained=True),
                ]
            ),
        )
        config: EveryVoiceConfig = EveryVoiceConfig(
            vocoder=vocoder_config,
            contact=self.contact,
            aligner=AlignerConfig(contact=self.contact),
            feature_prediction=FeaturePredictionConfig(contact=self.contact),
        )
        sox_effects = config.vocoder.preprocessing.source_data[0].sox_effects
        self.assertEqual(len(config.vocoder.preprocessing.source_data), 4)
        for d_other in config.vocoder.preprocessing.source_data[1:]:
            self.assertEqual(sox_effects, d_other.sox_effects)
            self.assertEqual(sox_effects[0], ["channels", "1"])

    def test_correct_number_typing(self):
        batch_size = 64.0
        config = EveryVoiceConfig(
            training=E2ETrainingConfig(batch_size=batch_size),
            contact=self.contact,
            aligner=AlignerConfig(contact=self.contact),
            feature_prediction=FeaturePredictionConfig(contact=self.contact),
            vocoder=VocoderConfig(contact=self.contact),
        )
        self.assertIsInstance(batch_size, float)
        self.assertEqual(config.training.batch_size, 64)
        self.assertIsInstance(config.feature_prediction.training.batch_size, int)


class LoadConfigTest(BasicTestCase):
    """Load configs that contains relative paths."""

    REL_DATA_DIR = Path(__file__).parent / "data" / "relative" / "config"
    DATASET_NAME: str = "relative"

    def validate_config_path(self, path: Path):
        """
        Helper method to validate a path once loaded by a config.
        """
        self.assertTrue(path.is_absolute(), msg=path)
        self.assertTrue(path.exists(), msg=path)

    def test_aligner_config(self):
        """Create a AlignerConfig which pydantic will validate for us."""
        config_path = self.REL_DATA_DIR / f"{ALIGNER_CONFIG_FILENAME_PREFIX}.yaml"
        with config_path.open("r", encoding="utf8") as f:
            pre_test = yaml.safe_load(f)
            self.assertFalse(
                Path(pre_test["path_to_preprocessing_config_file"]).is_absolute()
            )
            self.assertFalse(Path(pre_test["path_to_text_config_file"]).is_absolute())
            training = pre_test["training"]
            self.assertFalse(Path(training["logger"]["save_dir"]).is_absolute())
            self.assertFalse(Path(training["training_filelist"]).is_absolute())
            self.assertFalse(Path(training["validation_filelist"]).is_absolute())
        config = AlignerConfig.load_config_from_path(config_path)
        # print(config.model_dump_json(indent=2))
        self.assertTrue(isinstance(config, AlignerConfig))
        self.assertEqual(config.preprocessing.dataset, self.DATASET_NAME)
        self.validate_config_path(config.path_to_preprocessing_config_file)
        self.validate_config_path(config.path_to_text_config_file)
        self.validate_config_path(config.training.logger.save_dir)
        self.validate_config_path(config.training.training_filelist)
        self.validate_config_path(config.training.validation_filelist)

    def test_preprocessing_config(self):
        """Create a PreprocessingConfig which pydantic will validate for us."""
        config_path = self.REL_DATA_DIR / f"{PREPROCESSING_CONFIG_FILENAME_PREFIX}.yaml"
        with config_path.open("r", encoding="utf8") as f:
            pre_test = yaml.safe_load(f)
            self.assertFalse(Path(pre_test["save_dir"]).is_absolute())
            self.assertEqual(len(pre_test["source_data"]), 1)
            for data in pre_test["source_data"]:
                self.assertFalse(Path(data["data_dir"]).is_absolute())
                self.assertFalse(Path(data["filelist"]).is_absolute())
        config = PreprocessingConfig.load_config_from_path(config_path)
        # print(config.model_dump_json(indent=2))
        self.assertTrue(isinstance(config, PreprocessingConfig))
        self.assertEqual(config.dataset, self.DATASET_NAME)
        self.validate_config_path(config.save_dir)
        self.assertEqual(len(config.source_data), 1)
        for data in config.source_data:
            self.validate_config_path(data.data_dir)
            self.validate_config_path(data.filelist)

    def test_feature_prediction_config(self):
        """Create a FeaturePredictionConfig which pydantic will validate for us."""
        config_path = self.REL_DATA_DIR / f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml"
        with config_path.open("r", encoding="utf8") as f:
            pre_test = yaml.safe_load(f)
            self.assertFalse(
                Path(pre_test["path_to_preprocessing_config_file"]).is_absolute()
            )
            self.assertFalse(Path(pre_test["path_to_text_config_file"]).is_absolute())
            training = pre_test["training"]
            self.assertFalse(Path(training["logger"]["save_dir"]).is_absolute())
            self.assertFalse(Path(training["training_filelist"]).is_absolute())
            self.assertFalse(Path(training["validation_filelist"]).is_absolute())
        config = FeaturePredictionConfig.load_config_from_path(config_path)
        # print(config.model_dump_json(indent=2))
        self.assertEqual(config.preprocessing.dataset, self.DATASET_NAME)
        self.validate_config_path(config.path_to_text_config_file)
        self.validate_config_path(config.path_to_text_config_file)
        self.validate_config_path(config.training.logger.save_dir)
        self.validate_config_path(config.training.training_filelist)
        self.validate_config_path(config.training.validation_filelist)

    def test_vocoder_config(self):
        """Create a VocoderConfig which pydantic will validate for us."""
        config_path = self.REL_DATA_DIR / f"{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml"
        with config_path.open("r", encoding="utf8") as f:
            pre_test = yaml.safe_load(f)
            self.assertFalse(
                Path(pre_test["path_to_preprocessing_config_file"]).is_absolute()
            )
            training = pre_test["training"]
            self.assertFalse(Path(training["logger"]["save_dir"]).is_absolute())
            self.assertFalse(Path(training["training_filelist"]).is_absolute())
            self.assertFalse(Path(training["validation_filelist"]).is_absolute())
        config = VocoderConfig.load_config_from_path(config_path)
        # print(config.model_dump_json(indent=2))
        self.assertTrue(isinstance(config, VocoderConfig))
        self.assertEqual(config.preprocessing.dataset, self.DATASET_NAME)
        self.validate_config_path(config.path_to_preprocessing_config_file)
        self.validate_config_path(config.training.logger.save_dir)
        self.validate_config_path(config.training.training_filelist)
        self.validate_config_path(config.training.validation_filelist)

    def test_everyvoice_config(self):
        """Create a EveryVoiceConfig which pydantic will validate for us."""
        config_path = self.REL_DATA_DIR / f"{TEXT_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml"
        with config_path.open("r", encoding="utf8") as f:
            pre_test = yaml.safe_load(f)
            self.assertFalse(
                Path(pre_test["path_to_aligner_config_file"]).is_absolute()
            )
            self.assertFalse(
                Path(pre_test["path_to_feature_prediction_config_file"]).is_absolute()
            )
            self.assertFalse(
                Path(pre_test["path_to_vocoder_config_file"]).is_absolute()
            )
            training = pre_test["training"]
            self.assertFalse(Path(training["logger"]["save_dir"]).is_absolute())
            self.assertFalse(Path(training["training_filelist"]).is_absolute())
            self.assertFalse(Path(training["validation_filelist"]).is_absolute())
        config = EveryVoiceConfig.load_config_from_path(config_path)
        # print(config.model_dump_json(indent=2))
        self.assertTrue(isinstance(config, EveryVoiceConfig))
        self.assertEqual(
            config.feature_prediction.preprocessing.dataset, self.DATASET_NAME
        )
        self.validate_config_path(config.path_to_aligner_config_file)
        self.validate_config_path(config.path_to_feature_prediction_config_file)
        self.validate_config_path(config.path_to_vocoder_config_file)
        self.validate_config_path(config.training.logger.save_dir)
        self.validate_config_path(config.training.training_filelist)
        self.validate_config_path(config.training.validation_filelist)

    def test_absolute_path(self):
        """Load a config that has absolute paths."""
        with tempfile.TemporaryDirectory(
            prefix="test_absolute_path"
        ) as tempdir, init_context({"writing_config": Path(tempdir)}):
            tempdir = Path(tempdir).absolute()
            # Write preprocessing:
            preprocessing_config_path = tempdir / "aligner-preprocessing.json"
            _writer_helper(
                PreprocessingConfig(dataset=self.DATASET_NAME),
                preprocessing_config_path,
            )

            # Write text:
            text_config_path = tempdir / "aligner-text.json"
            _writer_helper(TextConfig(), text_config_path)

            # Write training:
            aligner_training_path = tempdir / "aligner-training.json"
            training = DFAlignerTrainingConfig(
                training_filelist=tempdir / "training_filelist.psv",
                validation_filelist=tempdir / "validation_filelist.psv",
            )
            (tempdir / training.logger.save_dir).mkdir(parents=True, exist_ok=True)
            print(tempdir, training.training_filelist)
            (training.training_filelist).touch(exist_ok=True)
            (training.validation_filelist).touch(exist_ok=True)
            _writer_helper(training, aligner_training_path)

            # Write model:
            aligner_model_path = tempdir / "aligner-model.json"
            _writer_helper(
                AlignerConfig(contact=self.contact).model, aligner_model_path
            )

            # Aligner Config
            aligner_config = AlignerConfig(
                contact=self.contact,
                path_to_model_config_file=aligner_model_path,
                path_to_preprocessing_config_file=preprocessing_config_path,
                path_to_text_config_file=text_config_path,
                path_to_training_config_file=aligner_training_path,
            )
            self.assertTrue(isinstance(aligner_config, AlignerConfig))
            aligner_config_path = tempdir / "aligner.json"
            _writer_helper(aligner_config, aligner_config_path)

            # Reload and validate
            config = AlignerConfig.load_config_from_path(aligner_config_path)
            self.assertTrue(isinstance(config, AlignerConfig))
            self.assertEqual(config.preprocessing.dataset, self.DATASET_NAME)
            self.validate_config_path(config.path_to_model_config_file)
            self.validate_config_path(config.path_to_preprocessing_config_file)
            self.validate_config_path(config.path_to_text_config_file)
            self.validate_config_path(config.path_to_training_config_file)
            self.validate_config_path(config.training.logger.save_dir)
            self.validate_config_path(config.training.training_filelist)
            self.validate_config_path(config.training.validation_filelist)

    def test_missing_path(self):
        """Load a config that is missing a partial config file."""
        with tempfile.TemporaryDirectory(
            prefix="test_missing_path"
        ) as tempdir, init_context({"writing_config": Path(tempdir)}):
            tempdir = Path(tempdir)
            _writer_helper(AudioConfig(), tempdir / "audio.json")
            config = PreprocessingConfig(
                path_to_audio_config_file=(tempdir / "audio.json")
            )
            self.assertTrue(isinstance(config.audio, AudioConfig))
            # Write shared:
            _writer_helper(
                PreprocessingConfig(dataset=self.DATASET_NAME),
                tempdir / "preprocessing.json",
            )
            _writer_helper(TextConfig(), tempdir / "text.json")
            _writer_helper(BaseTrainingConfig(), tempdir / "training.json")
            # Aligner Config
            _writer_helper(
                AlignerConfig(contact=self.contact).training,
                tempdir / "aligner-training.json",
            )
            _writer_helper(
                AlignerConfig(contact=self.contact).model,
                tempdir / "aligner-model.json",
            )
            aligner_config = AlignerConfig(
                contact=self.contact,
                path_to_model_config_file=tempdir / "aligner-model.json",
                path_to_preprocessing_config_file=tempdir / "preprocessing.json",
                path_to_text_config_file=tempdir / "text.json",
                path_to_training_config_file=tempdir / "aligner-training.json",
            )
            _writer_helper(aligner_config, tempdir / "aligner.json")
            self.assertTrue(isinstance(aligner_config, AlignerConfig))
            # Create the missing partial config file by deleting.
            # NOTE, we need the file to exists if we want to write its parent config to disk.
            (tempdir / "preprocessing.json").unlink()
            with self.assertRaises(ValidationError):
                config = AlignerConfig.load_config_from_path(tempdir / "aligner.json")


class BaseTrainingConfigTest(TestCase):
    """
    Validate BaseTrainingConfig
    """

    def test_ckpt_epochs_cannot_be_negative(self):
        """
        every_n_epochs aka ckpt_epochs must be None or non-negative.
        """
        config = BaseTrainingConfig(ckpt_epochs=None, ckpt_steps=None)
        self.assertEqual(config.ckpt_epochs, None)

        config = BaseTrainingConfig(ckpt_epochs=0, ckpt_steps=None)
        self.assertEqual(config.ckpt_epochs, 0)

        config = BaseTrainingConfig(ckpt_epochs=10, ckpt_steps=None)
        self.assertEqual(config.ckpt_epochs, 10)

        with self.assertRaises(ValueError):
            _ = BaseTrainingConfig(ckpt_epochs=-1, ckpt_steps=None)

    def test_ckpt_steps_cannot_be_negative(self):
        """
        every_n_train_steps aka ckpt_steps must be None or non-negative.
        """
        config = BaseTrainingConfig(ckpt_epochs=None, ckpt_steps=None)
        self.assertEqual(config.ckpt_steps, None)

        config = BaseTrainingConfig(ckpt_epochs=None, ckpt_steps=0)
        self.assertEqual(config.ckpt_steps, 0)

        config = BaseTrainingConfig(ckpt_epochs=None, ckpt_steps=10)
        self.assertEqual(config.ckpt_steps, 10)

        with self.assertRaises(ValueError):
            _ = BaseTrainingConfig(ckpt_epochs=None, ckpt_steps=-1)

    def test_mutually_exclusive_ckpt_options(self):
        """
        ckpt_epochs and ckpt_steps must be mutually exclusive.
        """
        config = BaseTrainingConfig()
        self.assertEqual(config.ckpt_epochs, 1)
        self.assertEqual(config.ckpt_steps, None)

        config = BaseTrainingConfig(ckpt_epochs=None, ckpt_steps=None)
        self.assertEqual(config.ckpt_epochs, None)
        self.assertEqual(config.ckpt_steps, None)

        config = BaseTrainingConfig(ckpt_epochs=7, ckpt_steps=None)
        self.assertEqual(config.ckpt_epochs, 7)
        self.assertEqual(config.ckpt_steps, None)

        config = BaseTrainingConfig(ckpt_epochs=None, ckpt_steps=11)
        self.assertEqual(config.ckpt_epochs, None)
        self.assertEqual(config.ckpt_steps, 11)

        with self.assertRaises(ValueError):
            _ = BaseTrainingConfig(
                ckpt_epochs=1,
                ckpt_steps=1,
            )


class HiFiGANModelConfigTest(TestCase):
    """
    This class should really be under HiFiGan.
    """

    def test_invalide_resblock(self):
        """
        Validate that we get a nice error message when the user provides a
        resblock that is not part of the enum's values.
        """
        from pydantic_core._pydantic_core import ValidationError

        with self.assertRaisesRegex(
            ValidationError,
            r"Input should be '1' or '2' \[type=enum, input_value='BAD', input_type=str\]",
        ):
            HiFiGANModelConfig(resblock="BAD")


class HiFiGANTrainingConfigTest(TestCase):
    """
    This class should really be under HiFiGan.
    """

    def test_invalid_gan_type(self):
        """
        Validate that we get a nice error message when the user provides a
        gan_type that is not part of the enum's values.
        """
        from pydantic_core._pydantic_core import ValidationError

        with self.assertRaisesRegex(
            ValidationError,
            r"Input should be 'original' or 'wgan' \[type=enum, input_value='BAD', input_type=str\]",
        ):
            HiFiGANTrainingConfig(gan_type="BAD")
