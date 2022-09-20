from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from config import CONFIGS
from dataloader import HiFiGANDataModule
from model.vocoder.hifigan import HiFiGAN

CONFIG = CONFIGS["base"]

TENSORBOARD_LOGGER = TensorBoardLogger(**CONFIG["training"]["logger"])

if CONFIG["training"]["strategy"] == "vocoder":
    TRAINER = Trainer(
        logger=TENSORBOARD_LOGGER,
        accelerator="auto",
        devices="auto",
        max_epochs=CONFIG["training"]["vocoder"]["max_epochs"],
    )
    VOCODER = HiFiGAN(CONFIG)
    data = HiFiGANDataModule(CONFIG)
    TENSORBOARD_LOGGER.log_hyperparams(CONFIG)
    TRAINER.fit(
        VOCODER,
        data,
        ckpt_path=CONFIG["training"]["vocoder"]["finetune_checkpoint"],
    )

if CONFIG["training"]["strategy"] == "feature_prediction":
    pass

if CONFIG["training"]["strategy"] == "e2e":
    # see https://github.com/Lightning-AI/lightning/issues/7447#issuecomment-835695726
    pass
