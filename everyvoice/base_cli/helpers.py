"""This is the location of basic cli commands that can be copied to new model cli's
    We use the merge-args to merge function signatures between the base functions described here
    and the model-specific ones defined in everyvoice/model/*/*/*/cli
    We want to do it this way to preserve the functionality from typer's command() decorator
    inferring information from the function signature while still keeping code DRY.
"""
import os
from enum import Enum
from pathlib import Path
from pprint import pformat
from typing import List, Optional, Union

from deepdiff import DeepDiff
from loguru import logger
from tqdm import tqdm

from everyvoice.model.aligner.config import DFAlignerConfig
from everyvoice.model.aligner.DeepForcedAligner.dfaligner.dataset import (
    AlignerDataModule,
)
from everyvoice.model.aligner.DeepForcedAligner.dfaligner.model import Aligner
from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.e2e.dataset import E2EDataModule
from everyvoice.model.e2e.model import EveryVoice
from everyvoice.model.feature_prediction.config import FastSpeech2Config
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.dataset import (
    FastSpeech2DataModule,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
    FastSpeech2,
)
from everyvoice.model.vocoder.config import HiFiGANConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.dataset import (
    HiFiGANDataModule,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import HiFiGAN


def load_config_base_command(
    model_config: Union[
        DFAlignerConfig, EveryVoiceConfig, FastSpeech2Config, HiFiGANConfig
    ],
    # Must include the above in model-specific command
    config_args: List[str],
    config_file: Path,
):
    from everyvoice.utils import update_config_from_cli_args

    config = model_config.load_config_from_path(config_file)

    config = update_config_from_cli_args(config_args, config)
    return config


def preprocess_base_command(
    model_config: Union[
        DFAlignerConfig, EveryVoiceConfig, FastSpeech2Config, HiFiGANConfig
    ],
    steps: List[str],
    # Must include the above in model-specific command
    config_args: List[str],
    config_file: Path,
    output_path: Optional[Path],
    cpus: Optional[int],
    overwrite: bool,
    debug: bool,
):
    from everyvoice.preprocessor import Preprocessor

    config = load_config_base_command(model_config, config_args, config_file)
    preprocessor = Preprocessor(config)
    if isinstance(config, FastSpeech2Config) and config.model.use_phonological_feats:
        steps.append("pfs")
    preprocessor.preprocess(
        output_path=output_path,
        cpus=cpus,
        overwrite=overwrite,
        to_process=steps,
        debug=debug,
    )
    return preprocessor, config, steps


def train_base_command(
    model_config: Union[
        DFAlignerConfig, EveryVoiceConfig, FastSpeech2Config, HiFiGANConfig
    ],
    data_module: Union[
        AlignerDataModule, E2EDataModule, FastSpeech2DataModule, HiFiGANDataModule
    ],
    model: Union[Aligner, EveryVoice, FastSpeech2, HiFiGAN],
    monitor: str,
    # Must include the above in model-specific command
    config_args: List[str],
    config_file: Path,
    accelerator: str,
    devices: str,
    nodes: int,
    strategy: str,
):
    config = load_config_base_command(model_config, config_args, config_file)
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
    tensorboard_logger = TensorBoardLogger(**(config.training.logger.model_dump()))
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
        max_steps=config.training.max_steps,
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
    # Train from Scratch
    if last_ckpt is None:
        model_obj = model(config)
        tensorboard_logger.log_hyperparams(config.model_dump())
        trainer.fit(model_obj, data)
    else:
        model_obj = model.load_from_checkpoint(last_ckpt)
        # Check if the trainer has changed (but ignore subdir since it is specific to the run)
        diff = DeepDiff(
            model_obj.config.training.model_dump(), config.training.model_dump()
        )
        training_config_diff = [
            item for item in diff["values_changed"].items() if "sub_dir" not in item[0]
        ]
        if training_config_diff:
            model_obj.config.training = config.training
            tensorboard_logger.log_hyperparams(config.model_dump())
            # Finetune from Checkpoint
            logger.warning(
                f"""Some of your training hyperparameters have changed from your checkpoint at '{last_ckpt}', so we will override your checkpoint hyperparameters.
                               Your training logs will start from epoch 0/step 0, but will still use the weights from your checkpoint. Values Changed: {pformat(training_config_diff)}
                            """
            )
            trainer.fit(model_obj, data)
        else:
            logger.info(f"Resuming from checkpoint '{last_ckpt}'")
            # Resume from checkpoint
            tensorboard_logger.log_hyperparams(config.model_dump())
            trainer.fit(model_obj, data, ckpt_path=last_ckpt)


def inference_base_command(name: Enum):
    pass
