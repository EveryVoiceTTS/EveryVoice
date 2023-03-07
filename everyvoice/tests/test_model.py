from unittest import TestCase, main

from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import HiFiGAN


class ModelTest(TestCase):
    """Basic test for models"""

    def setUp(self) -> None:
        self.config = EveryVoiceConfig.load_config_from_path()
        self.hifi_gan = HiFiGAN(self.config.vocoder)

    def test_hparams(self):
        self.assertEqual(self.config.vocoder, self.hifi_gan.hparams.config)
        self.assertEqual(self.config.vocoder, self.hifi_gan.config)


if __name__ == "__main__":
    main()
