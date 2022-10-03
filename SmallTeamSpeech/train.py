import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from config import CONFIGS
from dataloader import HiFiGANDataModule
from model.vocoder.hifigan import HiFiGAN

CONFIG = CONFIGS["lj"]

TENSORBOARD_LOGGER = TensorBoardLogger(**CONFIG["training"]["logger"])


if CONFIG["training"]["strategy"] == "vocoder":
    ckpt_callback = ModelCheckpoint(
        monitor="validation/mel_spec_error",
        mode="min",
        save_last=True,
        save_top_k=CONFIG["training"]["vocoder"]["save_top_k_ckpts"],
        every_n_train_steps=CONFIG["training"]["vocoder"]["ckpt_steps"],
        every_n_epochs=CONFIG["training"]["vocoder"]["ckpt_epochs"],
    )
    TRAINER = Trainer(
        logger=TENSORBOARD_LOGGER,
        accelerator="auto",
        devices="auto",
        max_epochs=CONFIG["training"]["vocoder"]["max_epochs"],
        callbacks=[ckpt_callback],
    )
    VOCODER = HiFiGAN(CONFIG)
    data = HiFiGANDataModule(CONFIG)
    last_ckpt = (
        CONFIG["training"]["vocoder"]["finetune_checkpoint"]
        if os.path.exists(CONFIG["training"]["vocoder"]["finetune_checkpoint"])
        else None
    )
    TENSORBOARD_LOGGER.log_hyperparams(CONFIG)
    TRAINER.fit(
        VOCODER,
        data,
        ckpt_path=last_ckpt,
    )

if CONFIG["training"]["strategy"] == "feature_prediction":
    pass

if CONFIG["training"]["strategy"] == "e2e":
    # see https://github.com/Lightning-AI/lightning/issues/7447#issuecomment-835695726
    pass
