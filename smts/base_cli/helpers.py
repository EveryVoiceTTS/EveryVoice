"""This is the location of basic cli commands that can be copied to new model cli's
    We use the merge-args to merge function signatures between the base functions described here
    and the model-specific ones defined in smts/model/*/*/*/cli
    We want to do it this way to preserve the functionality from typer's command() decorator
    inferring information from the function signature while still keeping code DRY.
"""
import os
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from loguru import logger
from tqdm import tqdm

from smts.model.aligner.config import DFAlignerConfig
from smts.model.aligner.DeepForcedAligner.dfaligner.dataset import AlignerDataModule
from smts.model.aligner.DeepForcedAligner.dfaligner.model import Aligner
from smts.model.e2e.config import SMTSConfig
from smts.model.e2e.dataset import E2EDataModule
from smts.model.e2e.model import SmallTeamSpeech
from smts.model.feature_prediction.config import FastSpeech2Config
from smts.model.feature_prediction.FastSpeech2_lightning.fs2.dataset import (
    FastSpeech2DataModule,
)
from smts.model.feature_prediction.FastSpeech2_lightning.fs2.model import FastSpeech2
from smts.model.vocoder.config import HiFiGANConfig
from smts.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.dataset import HiFiGANDataModule
from smts.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import HiFiGAN


def load_config_base_command(
    name: Enum,
    model_config: Union[DFAlignerConfig, SMTSConfig, FastSpeech2Config, HiFiGANConfig],
    configs,
    # Must include the above in model-specific command
    config_args: List[str],
    config_path: Path,
):
    from smts.utils import update_config_from_cli_args

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
    model_config: Union[DFAlignerConfig, SMTSConfig, FastSpeech2Config, HiFiGANConfig],
    configs,
    data,
    preprocess_categories,
    # Must include the above in model-specific command
    config_args: List[str],
    config_path: Path,
    output_path: Optional[Path],
    cpus: Optional[int],
    overwrite: bool,
    compute_stats=True,
):
    from smts.preprocessor import Preprocessor

    config = load_config_base_command(
        name, model_config, configs, config_args, config_path
    )
    to_preprocess = {
        f"process_{k}": k in data for k in preprocess_categories.__members__.keys()
    }
    preprocessor = Preprocessor(config)
    if not data:
        logger.info(
            f"No specific preprocessing data requested, processing everything from dataset '{name}'"
        )
        to_preprocess = {k: True for k in to_preprocess}
        if isinstance(config, FastSpeech2Config):
            to_preprocess["process_pfs"] = config.model.use_phonological_feats
    preprocessor.preprocess(
        output_path=output_path,
        compute_stats=compute_stats,
        cpus=cpus,
        overwrite=overwrite,
        **to_preprocess,
    )


def train_base_command(
    name: Enum,
    model_config: Union[DFAlignerConfig, SMTSConfig, FastSpeech2Config, HiFiGANConfig],
    configs,
    data_module: Union[
        AlignerDataModule, E2EDataModule, FastSpeech2DataModule, HiFiGANDataModule
    ],
    model: Union[Aligner, SmallTeamSpeech, FastSpeech2, HiFiGAN],
    monitor: str,
    # Must include the above in model-specific command
    config_args: List[str],
    config_path: Path,
    accelerator: str,
    devices: str,
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
    pbar.set_description("Loading SmallTeamSpeech modules")

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
