from pytorch_lightning import Trainer

from config import CONFIGS
from dataloader import HiFiGANDataModule
from model.vocoder.hifigan import HiFiGAN

CONFIG = CONFIGS["lj"]

if CONFIG["training"]["strategy"] == "vocoder":
    TRAINER = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=CONFIG["training"]["vocoder"]["max_epochs"],
        auto_scale_batch_size=True,
    )
    VOCODER = HiFiGAN(CONFIG)
    data = HiFiGANDataModule(CONFIG)
    TRAINER.tune(VOCODER, data)

if CONFIG["training"]["strategy"] == "feature_prediction":
    pass

if CONFIG["training"]["strategy"] == "e2e":
    # see https://github.com/Lightning-AI/lightning/issues/7447#issuecomment-835695726
    pass
