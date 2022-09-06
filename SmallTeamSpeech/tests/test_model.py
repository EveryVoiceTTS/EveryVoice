from unittest import TestCase

from config import BaseConfig
from model.vocoder.hifigan import HiFiGAN


class ModelTest(TestCase):
    """Basic test for models"""

    def setUp(self) -> None:
        self.hifi_gan = HiFiGAN(BaseConfig())

    def test_hparams(self):
        self.assertEqual(BaseConfig(), self.hifi_gan.hparams.config)
        self.assertEqual(BaseConfig(), self.hifi_gan.config)
