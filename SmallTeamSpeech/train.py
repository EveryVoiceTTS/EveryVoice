from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from config import CONFIGS
from dataloader import HiFiGANDataModule
from model.vocoder.hifigan import HiFiGAN

CONFIG = CONFIGS["base"]

TENSORBOARD_LOGGER = TensorBoardLogger(**CONFIG["training"]["logger"])


if CONFIG["training"]["strategy"] == "vocoder":
    ckpt_callback = ModelCheckpoint(
        monitor="validation/mel_spec_error",
        mode="min",
        save_top_k=CONFIG["training"]["vocoder"]["save_top_k_ckpts"],
        # every_n_train_steps=CONFIG["training"]["vocoder"]["ckpt_steps"],
        every_n_epochs=100,
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
