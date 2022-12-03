import os
from enum import Enum
from pathlib import Path
from typing import List

import typer
from loguru import logger

from smts.model.e2e.config import CONFIGS, SMTSConfig

app = typer.Typer(pretty_exceptions_show_locals=False)

_config_keys = {k: k for k in CONFIGS.keys()}

CONFIGS_ENUM = Enum("CONFIGS", _config_keys)  # type: ignore


@app.command()
def train(
    name: CONFIGS_ENUM,
    accelerator: str = typer.Option("auto", "--accelerator", "-a"),
    devices: str = typer.Option("auto", "--devices", "-d"),
    strategy: str = typer.Option(None),
    config_args: List[str] = typer.Option(None, "--config", "-c"),
    config_path: Path = typer.Option(None, exists=True, dir_okay=False, file_okay=True),
):
    from smts.model.e2e.dataset import E2EDataModule
    from smts.model.e2e.model import SmallTeamSpeech
    from smts.utils import update_config_from_cli_args, update_config_from_path

    original_config = SMTSConfig.load_config_from_path(CONFIGS[name.value])
    config = update_config_from_cli_args(config_args, original_config)
    config = update_config_from_path(config_path, config)

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    tensorboard_logger = TensorBoardLogger(**(config.training.logger.dict()))
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger.info("Starting joint training of end-to-end model.")
    ckpt_callback = ModelCheckpoint(
        monitor="training/total_loss",
        mode="min",
        save_last=True,
        save_top_k=config.training.save_top_k_ckpts,
        every_n_train_steps=config.training.ckpt_steps,
        every_n_epochs=config.training.ckpt_epochs,
    )
    trainer = Trainer(
        gradient_clip_val=1.0,
        logger=tensorboard_logger,
        accelerator=accelerator,
        devices=devices,
        max_epochs=config.training.max_epochs,
        callbacks=[ckpt_callback, lr_monitor],
        strategy=strategy,
        detect_anomaly=False,  # used for debugging, but triples training time
    )
    model = SmallTeamSpeech(config)
    data = E2EDataModule(config)
    last_ckpt = (
        config.training.finetune_checkpoint
        if config.training.finetune_checkpoint is not None
        and os.path.exists(config.training.finetune_checkpoint)
        else None
    )
    tensorboard_logger.log_hyperparams(config.dict())
    trainer.fit(model, data, ckpt_path=last_ckpt)
