from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

from config import CONFIGS
from dataloader import HiFiGANDataModule
from model.vocoder.hifigan import HiFiGAN

CONFIG = CONFIGS["base"]

MLF_LOGGER = MLFlowLogger(**CONFIG["training"]["logger"])

if CONFIG["training"]["strategy"] == "vocoder":
    TRAINER = Trainer(
        logger=MLF_LOGGER,
        accelerator="gpu",
        devices=1,
        max_epochs=CONFIG["training"]["vocoder"]["max_epochs"],
    )
    VOCODER = HiFiGAN(CONFIG)
    data = HiFiGANDataModule(CONFIG)
    TRAINER.fit(VOCODER, data)

if CONFIG["training"]["strategy"] == "feature_prediction":
    pass

if CONFIG["training"]["strategy"] == "e2e":
    # see https://github.com/Lightning-AI/lightning/issues/7447#issuecomment-835695726
    pass
