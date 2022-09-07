from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

from SmallTeamSpeech.config import CONFIGS
from SmallTeamSpeech.model.vocoder.hifigan import HiFiGAN

CONFIG = CONFIGS["base"]

MLF_LOGGER = MLFlowLogger(**CONFIG["training"]["logger"])
TRAINER = Trainer(logger=MLF_LOGGER)

if CONFIG["training"]["strategy"] == "vocoder":
    VOCODER = HiFiGAN(CONFIG)
    # TODO: load vocoder data
    train_dataloader = None
    val_dataloader = None
    TRAINER.fit(VOCODER, train_dataloader, val_dataloader)

if CONFIG["training"]["strategy"] == "feature_prediction":
    pass

if CONFIG["training"]["strategy"] == "e2e":
    # see https://github.com/Lightning-AI/lightning/issues/7447#issuecomment-835695726
    pass
