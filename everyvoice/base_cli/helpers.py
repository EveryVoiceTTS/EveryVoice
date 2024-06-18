"""This is the location of basic cli commands that can be copied to new model cli's
    We use the merge-args to merge function signatures between the base functions described here
    and the model-specific ones defined in everyvoice/model/*/*/*/cli
    We want to do it this way to preserve the functionality from typer's command() decorator
    inferring information from the function signature while still keeping code DRY.
"""

import json
import os
import tempfile
import textwrap
from enum import Enum
from pathlib import Path
from pprint import pformat
from typing import List, Optional, Union

import yaml
from deepdiff import DeepDiff
from loguru import logger
from pydantic import ValidationError
from tqdm import tqdm

from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel
from everyvoice.exceptions import InvalidConfiguration
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
        type[DFAlignerConfig],
        type[EveryVoiceConfig],
        type[FastSpeech2Config],
        type[HiFiGANConfig],
    ],
    # Must include the above in model-specific command
    config_args: List[str],
    config_file: Path,
):
    from everyvoice.utils import update_config_from_cli_args

    try:
        config = model_config.load_config_from_path(config_file)
    except ValidationError as error:
        # NOTE: To trigger this error handling code from the command line:
        #   `everyvoice preprocess  config/everyvoice-aligner.yaml`
        import sys

        for config_type in (
            DFAlignerConfig,
            EveryVoiceConfig,
            FastSpeech2Config,
            HiFiGANConfig,
        ):
            try:
                config = config_type.load_config_from_path(  # type: ignore[attr-defined]
                    config_file
                )
                logger.error(
                    f"We are expecting a {model_config.__name__} but it looks like you provided a {config_type.__name__}"
                )
                sys.exit(1)
            except ValidationError:
                pass

        logger.error(f"there was a problem with your config file:\n{error}")
        sys.exit(1)

    config = update_config_from_cli_args(config_args, config)
    return config


def preprocess_base_command(
    model_config: Union[
        type[DFAlignerConfig],
        type[EveryVoiceConfig],
        type[FastSpeech2Config],
        type[HiFiGANConfig],
    ],
    steps: List[str],
    # Must include the above in model-specific command
    config_args: List[str],
    config_file: Path,
    cpus: Optional[int],
    overwrite: bool,
    debug: bool,
):
    from everyvoice.preprocessor import Preprocessor

    config = load_config_base_command(model_config, config_args, config_file)
    preprocessor = Preprocessor(config)
    if (
        isinstance(config, FastSpeech2Config)
        and config.model.target_text_representation_level
        == TargetTrainingTextRepresentationLevel.phonological_features
    ):
        steps.append("pfs")
    preprocessor.preprocess(
        cpus=cpus,
        overwrite=overwrite,
        to_process=steps,
        debug=debug,
    )
    return preprocessor, config, steps


def save_configuration_to_log_dir(
    config: Union[DFAlignerConfig, EveryVoiceConfig, FastSpeech2Config, HiFiGANConfig]
):
    """
    Adds a logging file to the module's logger.
    Records to hparams.yaml the function's configuration.
    """
    log_dir = config.training.logger.save_dir / config.training.logger.name
    log_dir.mkdir(exist_ok=True, parents=True)
    logger.add(log_dir / "log")

    hyperparameters_log = log_dir / "hparams.yaml"
    hyperparameters_log.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Configuration\n{config.model_dump_json(indent=3)}"
    )  # Once to be logged
    with hyperparameters_log.open(mode="w", encoding="UTF-8") as cout:
        output = json.loads(config.model_dump_json())
        yaml.dump(output, stream=cout)


def train_base_command(
    model_config: Union[
        type[DFAlignerConfig],
        type[EveryVoiceConfig],
        type[FastSpeech2Config],
        type[HiFiGANConfig],
    ],
    data_module: Union[
        type[AlignerDataModule],
        type[E2EDataModule],
        type[FastSpeech2DataModule],
        type[HiFiGANDataModule],
    ],
    model: Union[type[Aligner], type[EveryVoice], type[FastSpeech2], type[HiFiGAN]],
    monitor: str,
    # Must include the above in model-specific command
    config_args: List[str],
    config_file: Path,
    accelerator: str,
    devices: str,
    nodes: int,
    strategy: str,
    gradient_clip_val: float | None,
    model_kwargs={},
):
    from everyvoice.base_cli.callback import ResetValidationDataloaderCallback

    config = load_config_base_command(model_config, config_args, config_file)

    save_configuration_to_log_dir(config)

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
    tensorboard_logger = TensorBoardLogger(
        **{
            **(config.training.logger.model_dump(exclude={"sub_dir_callable": True})),
            **{"sub_dir": config.training.logger.sub_dir},
        }
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger.info("Starting training.")
    # This callback will always save the last checkpoint
    # regardless of its performance.
    last_ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        every_n_train_steps=config.training.ckpt_steps,
        every_n_epochs=config.training.ckpt_epochs,
        enable_version_counter=True,
    )
    # This callback will only save the top-k checkpoints
    # based on minimization of the monitored loss
    monitored_ckpt_callback = ModelCheckpoint(
        monitor=monitor,
        mode="min",
        save_top_k=config.training.save_top_k_ckpts,
        every_n_train_steps=config.training.ckpt_steps,
        every_n_epochs=config.training.ckpt_epochs,
        enable_version_counter=False,
    )
    trainer = Trainer(
        logger=tensorboard_logger,
        accelerator=accelerator,
        devices=devices,
        max_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
        val_check_interval=config.training.val_check_interval,
        callbacks=[
            monitored_ckpt_callback,
            last_ckpt_callback,
            lr_monitor,
            ResetValidationDataloaderCallback(),
        ],
        strategy=strategy,
        num_nodes=nodes,
        detect_anomaly=False,  # used for debugging, but triples training time
        gradient_clip_val=gradient_clip_val,
    )
    data = data_module(config)
    last_ckpt = (
        config.training.finetune_checkpoint
        if config.training.finetune_checkpoint is not None
        and os.path.exists(config.training.finetune_checkpoint)
        else None
    )
    # Train from Scratch
    if last_ckpt is None:
        model_obj = model(config, **model_kwargs)
        logger.info(f"Model's architecture\n{model_obj}")
        tensorboard_logger.log_hyperparams(config.model_dump())
        trainer.fit(model_obj, data)
    else:
        model_obj = model.load_from_checkpoint(last_ckpt)
        logger.info(f"Model's architecture\n{model_obj}")
        # Check if the trainer has changed (but ignore subdir since it is specific to the run)
        if isinstance(model_obj, EveryVoice) or isinstance(config, EveryVoiceConfig):
            optimizer_diff = DeepDiff((), ())
            model_diff = DeepDiff((), ())
        else:
            optimizer_diff = DeepDiff(
                model_obj.config.training.optimizer.model_dump(),
                config.training.optimizer.model_dump(),  # type : ignore[reportAttributeAccessIssue]
            )
            model_diff = DeepDiff(
                model_obj.config.model.model_dump(),
                config.model.model_dump(),
            )
        model_config_diff = []
        optimizer_config_diff = []
        if "values_changed" in model_diff:
            model_config_diff += list(model_diff["values_changed"].items())
        if "types_changed" in model_diff:
            model_config_diff += list(model_diff["types_changed"].items())
        if "values_changed" in optimizer_diff:
            optimizer_config_diff += list(optimizer_diff["values_changed"].items())
        if "types_changed" in optimizer_diff:
            optimizer_config_diff += list(optimizer_diff["types_changed"].items())
        if model_config_diff:
            raise InvalidConfiguration(
                textwrap.dedent(
                    f"""
                    Sorry, you are a trying to fine-tune a model with a different architecture defined in your configuration than was used during pre-training.

                    Please fix your configuration or use a different model.

                    Values Changed: {pformat(model_config_diff)}
                    """
                )
            )
        # If optimizer configuration is different, start training with updated optimizer hyperparameters
        # We need to override the model object's configuration with the current one.
        # This assumes that the model and optimizer configurations haven't changed since they
        # would be caught by the previous checks. TODO: We should also check for certain changes to
        # the text configuration, since certain changes would cause an input space mismatch.
        # FIXME: Cannot assign member "config" for type "HiFiGAN"
        model_obj.config = config
        if hasattr(model_obj, "update_config_settings"):
            model_obj.update_config_settings()
        tensorboard_logger.log_hyperparams(config.model_dump())
        if optimizer_config_diff:
            # Finetune from Checkpoint
            logger.warning(
                textwrap.dedent(
                    f"""
                    Some of your optimizer hyperparameters have changed from your checkpoint at '{last_ckpt}',
                    so we will override your checkpoint hyperparameters and restart the optimizer.

                    Your training logs will start from epoch 0/step 0, but will still use the weights from your checkpoint.

                    Values Changed: {pformat(optimizer_config_diff)}
                    """
                )
            )
            # This will only use the weights in the model_obj, the optimizer and current epoch etc will be restarted using
            # the configuration in model_obj.config, see https://github.com/Lightning-AI/pytorch-lightning/issues/5339
            trainer.fit(model_obj, data)
        else:
            import torch

            logger.info(f"Resuming from checkpoint '{last_ckpt}'")
            # We need to create a temporary checkpoint with torch.save because on_save_checkpoint
            # removes all paths for checkpoint portability. However, some paths, like "vocoder_path"
            # should be still accessible when training is resumed.
            new_config_with_paths = model_obj.config.model_dump(mode="json")
            old_ckpt = torch.load(last_ckpt, map_location=torch.device("cpu"))
            old_ckpt["hyper_parameters"]["config"] = new_config_with_paths
            # TODO: check if we need to do the same thing with stats and any thing else registered on the model
            with tempfile.NamedTemporaryFile() as tmp:
                torch.save(old_ckpt, tmp.name)
                # This will resume the weights, optimizer, and steps from last_ckpt, see https://github.com/Lightning-AI/pytorch-lightning/issues/5339
                trainer.fit(model_obj, data, ckpt_path=tmp.name)


def inference_base_command(_name: Enum):
    pass
