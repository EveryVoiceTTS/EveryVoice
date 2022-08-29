from unittest import TestCase, main
from config.base_config import BaseConfig
from config.base_config import (
    BASE_PREPROCESSING_HPARAMS,
    BASE_MODEL_HPARAMS,
    BASE_TRAINING_HPARAMS,
)


class ConfigTest(TestCase):
    """Basic test for hyperparameter configuration"""

    def setUp(self) -> None:
        pass

    def test_is_dict(self):
        base_config = BaseConfig()
        self.assertIsInstance(dict(base_config), dict)
        self.assertIn("model", base_config.keys())
        self.assertIn("training", base_config.keys())
        self.assertIn("preprocessing", base_config.keys())
        self.assertEqual(BASE_PREPROCESSING_HPARAMS, base_config["preprocessing"])
        self.assertEqual(BASE_MODEL_HPARAMS, base_config["model"])
        self.assertEqual(BASE_TRAINING_HPARAMS, base_config["training"])

    def test_changes(self):
        lj_config = BaseConfig(model={"max_seq_len": 1200})
        self.assertEqual(BASE_PREPROCESSING_HPARAMS, lj_config["preprocessing"])
        self.assertNotEqual(BASE_MODEL_HPARAMS, lj_config["model"])
        self.assertEqual(BASE_TRAINING_HPARAMS, lj_config["training"])
        self.assertEqual(lj_config)
