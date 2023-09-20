#!/usr/bin/env python

import json
import tempfile
import time
from pathlib import Path
from typing import Callable
from unittest import TestCase, main

import yaml

from everyvoice import exceptions
from everyvoice.config.preprocessing_config import Dataset, PreprocessingConfig
from everyvoice.config.shared_types import LoggerConfig
from everyvoice.config.text_config import TextConfig
from everyvoice.model.aligner.config import AlignerConfig
from everyvoice.model.e2e.config import E2ETrainingConfig, EveryVoiceConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.utils import (
    expand_config_string_syntax,
    load_config_from_json_or_yaml_path,
    lower,
    nfc_normalize,
)


class ConfigTest(TestCase):
    """Basic test for hyperparameter configuration"""

    data_dir = Path(__file__).parent / "data"

    def setUp(self) -> None:
        self.config = EveryVoiceConfig()

    def test_from_object(self):
        """Test from object"""
        config_default = EveryVoiceConfig()
        config_declared = EveryVoiceConfig(
            aligner=AlignerConfig(),
            feature_prediction=FeaturePredictionConfig(),
            vocoder=VocoderConfig(),
            training=E2ETrainingConfig(),
        )
        config_32 = EveryVoiceConfig(
            aligner=AlignerConfig(),
            feature_prediction=FeaturePredictionConfig(),
            vocoder=VocoderConfig(),
            training=E2ETrainingConfig(batch_size=32),
        )
        self.assertEqual(config_default.training.batch_size, 16)
        self.assertEqual(config_declared.training.batch_size, 16)
        self.assertEqual(config_32.training.batch_size, 32)

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
            text=TextConfig(cleaners=["everyvoice.utils.lower"])
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
        base_config = EveryVoiceConfig()
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
            {"feature_prediction": {"text": {"symbols": {"pad": "FOO"}}}}
        )
        self.assertEqual(self.config.feature_prediction.text.symbols.pad, "FOO")
        self.config.update_config(
            {"feature_prediction": {"text": {"cleaners": ["everyvoice.utils.lower"]}}}
        )
        self.assertEqual(self.config.feature_prediction.text.cleaners, [lower])
        self.assertEqual(self.config.feature_prediction.text.symbols.pad, "FOO")

    def test_load_empty_config(self):
        with tempfile.NamedTemporaryFile(prefix="test", mode="w", suffix=".yaml") as tf:
            tf.write(" ")
            with self.assertRaises(exceptions.InvalidConfiguration):
                load_config_from_json_or_yaml_path(Path(tf.name))

    def test_change_with_indices(self):
        """Text the --config-args can also work with arrays"""
        config = FeaturePredictionConfig()
        config.update_config(
            {
                "preprocessing": {
                    "source_data": {"0": {"filelist": "/foo/bar/filelist.psv"}}
                }
            }
        )
        self.assertEqual(
            str(config.preprocessing.source_data[0].filelist), "/foo/bar/filelist.psv"
        )

    def test_shared_sox(self):
        """Test that the shared sox config is correct"""
        vocoder_config = VocoderConfig(
            preprocessing=PreprocessingConfig(
                source_data=[Dataset(), Dataset(), Dataset(), Dataset()]
            )
        )
        config: EveryVoiceConfig = EveryVoiceConfig(vocoder=vocoder_config)
        sox_effects = config.vocoder.preprocessing.source_data[0].sox_effects
        self.assertEqual(len(config.vocoder.preprocessing.source_data), 4)
        for d_other in config.vocoder.preprocessing.source_data[1:]:
            self.assertEqual(sox_effects, d_other.sox_effects)
            self.assertEqual(sox_effects[0], ["channels", "1"])

    def test_correct_number_typing(self):
        batch_size = 64.0
        config = EveryVoiceConfig(training=E2ETrainingConfig(batch_size=batch_size))
        self.assertIsInstance(batch_size, float)
        self.assertEqual(config.training.batch_size, 64)
        self.assertIsInstance(config.feature_prediction.training.batch_size, int)


if __name__ == "__main__":
    main()
