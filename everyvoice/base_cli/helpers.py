"""This is the location of basic cli commands that can be copied to new model cli's
    We use the merge-args to merge function signatures between the base functions described here
    and the model-specific ones defined in everyvoice/model/*/*/*/cli
    We want to do it this way to preserve the functionality from typer's command() decorator
    inferring information from the function signature while still keeping code DRY.
"""
import os
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from loguru import logger
from tqdm import tqdm

from everyvoice.model.aligner.config import DFAlignerConfig
from everyvoice.model.aligner.DeepForcedAligner.dfaligner.dataset import AlignerDataModule
from everyvoice.model.aligner.DeepForcedAligner.dfaligner.model import Aligner
from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.e2e.dataset import E2EDataModule
from everyvoice.model.e2e.model import EveryVoice
from everyvoice.model.feature_prediction.config import FastSpeech2Config
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.dataset import (
    FastSpeech2DataModule,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import FastSpeech2
from everyvoice.model.vocoder.config import HiFiGANConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.dataset import HiFiGANDataModule
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import HiFiGAN


def load_config_base_command(
    name: Enum,
    model_config: Union[DFAlignerConfig, EveryVoiceConfig, FastSpeech2Config, HiFiGANConfig],
    configs,
    # Must include the above in model-specific command
    config_args: List[str],
    config_path: Path,
):
    from everyvoice.utils import update_config_from_cli_args

    if config_path:
        config = model_config.load_config_from_path(config_path)
    elif name:
        config = model_config.load_config_from_path(configs[name.value])
    else:
        logger.error(
            "You must either choose a <NAME> of a preconfigured dataset, or provide a <CONFIG_PATH> to a preprocessing configuration file."
        )
        exit()

    config = update_config_from_cli_args(config_args, config)
    return config


def preprocess_base_command(
    name: Enum,
    model_config: Union[DFAlignerConfig, EveryVoiceConfig, FastSpeech2Config, HiFiGANConfig],
    configs,
    data,
    preprocess_categories,
    # Must include the above in model-specific command
    config_args: List[str],
    config_path: Path,
    output_path: Optional[Path],
    cpus: Optional[int],
    overwrite: bool,
    debug: bool,
):
    from everyvoice.preprocessor import Preprocessor

    config = load_config_base_command(
        name, model_config, configs, config_args, config_path
    )
    to_process = [x.name for x in data]
    preprocessor = Preprocessor(config)
    if not data:
        logger.info(
            f"No specific preprocessing data requested, processing everything from dataset '{name}'"
        )
        to_process = list(preprocess_categories.__members__.keys())
        if (
            isinstance(config, FastSpeech2Config)
            and config.model.use_phonological_feats
        ):
            to_process.append("pfs")
    preprocessor.preprocess(
        output_path=output_path,
        cpus=cpus,
        overwrite=overwrite,
        to_process=to_process,
        debug=debug,
    )
    return preprocessor, config, to_process


def train_base_command(
    name: Enum,
    model_config: Union[DFAlignerConfig, EveryVoiceConfig, FastSpeech2Config, HiFiGANConfig],
    configs,
    data_module: Union[
        AlignerDataModule, E2EDataModule, FastSpeech2DataModule, HiFiGANDataModule
    ],
    model: Union[Aligner, EveryVoice, FastSpeech2, HiFiGAN],
    monitor: str,
    # Must include the above in model-specific command
    config_args: List[str],
    config_path: Path,
    accelerator: str,
    devices: str,
    nodes: int,
    strategy: str,
):
    config = load_config_base_command(
        name, model_config, configs, config_args, config_path
    )
    logger.info("Loading modules for training...")
    pbar = tqdm(range(4))
    pbar.set_description("Loading pytorch and friends")
    from pytorch_lightning import Trainer

    pbar.update()
    pbar.refresh()
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

    pbar.update()
    pbar.refresh()
    from pytorch_lightning.loggers import TensorBoardLogger

    pbar.update()
    pbar.refresh()
    pbar.set_description("Loading EveryVoice modules")

    pbar.update()
    pbar.refresh()
    tensorboard_logger = TensorBoardLogger(**(config.training.logger.dict()))
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger.info("Starting training.")
    ckpt_callback = ModelCheckpoint(
        monitor=monitor,
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
        num_nodes=nodes,
        detect_anomaly=False,  # used for debugging, but triples training time
    )
    model_obj = model(config)
    data = data_module(config)  # type: ignore
    last_ckpt = (
        config.training.finetune_checkpoint
        if config.training.finetune_checkpoint is not None
        and os.path.exists(config.training.finetune_checkpoint)
        else None
    )
    tensorboard_logger.log_hyperparams(config.dict())
    trainer.fit(model_obj, data, ckpt_path=last_ckpt)


def inference_base_command(name: Enum):
    pass
