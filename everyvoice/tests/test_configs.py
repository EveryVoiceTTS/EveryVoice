#!/usr/bin/env python

import json
from pathlib import Path
from unittest import TestCase, main

import yaml

from everyvoice.config.preprocessing_config import Dataset, PreprocessingConfig
from everyvoice.model.aligner.config import AlignerConfig
from everyvoice.model.e2e.config import E2ETrainingConfig, EveryVoiceConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.utils import expand_config_string_syntax, lower


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
