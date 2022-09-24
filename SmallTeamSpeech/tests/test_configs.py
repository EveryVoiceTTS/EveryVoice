import math
from unittest import TestCase

from config import CONFIGS
from config.base_config import (
    BASE_MODEL_HPARAMS,
    BASE_PREPROCESSING_HPARAMS,
    BASE_TRAINING_HPARAMS,
    BaseConfig,
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
        with self.assertRaises(ValueError):
            BaseConfig("This isn't a dictionary")

    def test_upsample(self):
        """Because the vocoder is set up to upsample"""
        for config in CONFIGS.values():
            # check that same number of kernels and kernel sizes exist
            sampling_rate = config["preprocessing"]["audio"]["input_sampling_rate"]
            upsampled_sampling_rate = config["preprocessing"]["audio"][
                "output_sampling_rate"
            ]
            self.assertEqual(
                len(config["model"]["vocoder"]["upsample_kernel_sizes"]),
                len(config["model"]["vocoder"]["upsample_rates"]),
            )
            # check that kernel sizes are not less than upsample rates, and are evenly divisible
            for i, upsample_kernel in enumerate(
                config["model"]["vocoder"]["upsample_kernel_sizes"]
            ):
                upsample_rate = config["model"]["vocoder"]["upsample_rates"][i]
                self.assertTrue(upsample_kernel >= upsample_rate)
                self.assertEqual(upsample_kernel % upsample_rate, 0)
            # check that upsampling rate is even multiple of target sampling rate
            self.assertEqual(upsampled_sampling_rate % sampling_rate, 0)
            # check that upsampling hop size is equal to product of upsample rates
            upsampling_hop_size = (upsampled_sampling_rate // sampling_rate) * config[
                "preprocessing"
            ]["audio"]["fft_hop_frames"]
            self.assertEqual(
                upsampling_hop_size,
                math.prod(config["model"]["vocoder"]["upsample_rates"]),
            )
            # check that segment size is divisible by product of upsample rates
            self.assertEqual(
                config["preprocessing"]["audio"]["vocoder_segment_size"]
                % math.prod(config["model"]["vocoder"]["upsample_rates"]),
                0,
            )
