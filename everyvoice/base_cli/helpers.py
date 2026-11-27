"""This is the location of basic cli commands that can be copied to new model cli's
We use the merge-args to merge function signatures between the base functions described here
and the model-specific ones defined in everyvoice/model/*/*/*/cli
We want to do it this way to preserve the functionality from typer's command() decorator
inferring information from the function signature while still keeping code DRY.
"""

import os
import sys
import textwrap
from pathlib import Path
from pprint import pformat
from typing import Any

from deepdiff import DeepDiff
from pydantic import ValidationError

from everyvoice import logger
from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel
from everyvoice.exceptions import InvalidConfiguration
from everyvoice.model.e2e.config import StyleTTS2Config
from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.lightning import (
    StyleTTS2,
    StyleTTS2DataModule,
)
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
from everyvoice.utils import update_config_from_cli_args

MODEL_CONFIGS = [FastSpeech2Config, HiFiGANConfig]


def load_unknown_config(
    config_file: Path,
):
    """This helper is for loading a config file that is unknown.
        ValidationErrors are passed until there is a succesful one.

    Args:
        config_file (Path): path to a configuration file
    Returns:
        config (PreprocessingConfig | FastSpeech2Config | HiFiGANConfig): the loaded config
    """
    from everyvoice.config.preprocessing_config import PreprocessingConfig

    errors = {}
    for config_type in (
        PreprocessingConfig,
        FastSpeech2Config,
        StyleTTS2Config,
        HiFiGANConfig,
    ):
        try:
            return config_type.load_config_from_path(config_file)  # type: ignore
        except ValidationError as e:
            error_count = e.error_count()
            errors[error_count] = str(e)
    min_error = min(errors.keys())
    raise ValueError(
        f"The config file at {config_file} was determined to not be any of the following. Please pass a valid EveryVoice model or shared data configuration file. {errors[min_error]}"
    )


def load_config_base_command(
    model_config: type[StyleTTS2Config] | type[FastSpeech2Config] | type[HiFiGANConfig],
    # Must include the above in model-specific command
    config_args: list[str],
    config_file: Path,
):

    try:
        config = model_config.load_config_from_path(config_file)
    except ValidationError as error:
        # NOTE: To trigger this error handling code from the command line,
        # provide a mismatched config file, e.g.:
        #   `everyvoice preprocess  config/everyvoice-spec-to-wav.yaml`
        import sys

        for config_type in (
            StyleTTS2Config,
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

    update_config_from_cli_args(config_args, config)
    return config


def preprocess_base_command(
    model_config: type[StyleTTS2Config] | type[FastSpeech2Config] | type[HiFiGANConfig],
    steps: list[str],
    # Must include the above in model-specific command
    config_args: list[str],
    config_file: Path,
    cpus: int,
    overwrite: bool,
    debug: bool,
):
    from everyvoice.preprocessor import Preprocessor

    config = load_config_base_command(model_config, config_args, config_file)
    preprocessor = Preprocessor(config)
    if (
        (isinstance(config, (FastSpeech2Config, StyleTTS2Config)))
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
    config: StyleTTS2Config | FastSpeech2Config | HiFiGANConfig,
):
    """
    Adds a logging file to the module's logger.
    Records to hparams.yaml the function's configuration.
    """
    import json

    import yaml

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
    model_config: type[StyleTTS2Config] | type[FastSpeech2Config] | type[HiFiGANConfig],
    data_module: (
        type[StyleTTS2DataModule]
        | type[FastSpeech2DataModule]
        | type[HiFiGANDataModule]
    ),
    model: type[StyleTTS2] | type[FastSpeech2] | type[HiFiGAN],
    monitor: str,
    # Must include the above in model-specific command
    config_args: list[str],
    config_file: Path,
    accelerator: str,
    devices: str,
    nodes: int,
    strategy: str,
    gradient_clip_val: float | None,
    auto_resume: bool = False,
    model_kwargs={},
):
    from tqdm import tqdm

    from everyvoice.base_cli.callback import ResetValidationDataloaderCallback
    from everyvoice.base_cli.checkpoint_io import atomic_checkpoint_io
    from everyvoice.base_cli.slurm import (
        resolve_resume_ckpt,
        save_requeue_ckpt_to_last,
    )

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
    from pytorch_lightning.plugins.io import TorchCheckpointIO

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
    # The checkpoint the "last" ModelCheckpoint saves, in the checkpoints dir
    # Lightning's ModelCheckpoint resolves from the logger
    last_ckpt = (
        Path(config.training.logger.save_dir)
        / config.training.logger.name
        / config.training.logger.version
        / "checkpoints"
        / "last.ckpt"
    )
    resume_ckpt = resolve_resume_ckpt(
        auto_resume, last_ckpt, config.training.finetune_checkpoint
    )
    logger.info("Starting training.")
    # This callback will always save the last checkpoint
    # regardless of its performance.
    last_ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        every_n_train_steps=config.training.ckpt_steps,
        every_n_epochs=config.training.ckpt_epochs,
        enable_version_counter=False,
        save_on_train_epoch_end=True,
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
        plugins=[atomic_checkpoint_io(TorchCheckpointIO)],
    )
    save_requeue_ckpt_to_last(trainer, last_ckpt_callback)
    data = data_module(config)

    # Continue this experiment from its own last checkpoint
    if resume_ckpt is not None:
        model_obj = model(config, **model_kwargs)
        tensorboard_logger.log_hyperparams(config.model_dump())
        fit_from_full_state(trainer, model_obj, data, resume_ckpt, config)
        return

    finetune_ckpt = config.training.finetune_checkpoint
    finetune_mode = config.training.finetune_mode
    if finetune_ckpt is not None and not os.path.exists(finetune_ckpt):
        logger.error(f"finetune_checkpoint '{finetune_ckpt}' does not exist.")
        sys.exit(1)
    # Train from Scratch
    if finetune_ckpt is None:
        if finetune_mode == "vocoder-matching":
            logger.error(
                "training.finetune_mode: vocoder-matching starts from a trained vocoder: "
                "set training.finetune_checkpoint to its checkpoint."
            )
            sys.exit(1)
        model_obj = model(config, **model_kwargs)
        logger.info(f"Model's architecture\n{model_obj}")
        tensorboard_logger.log_hyperparams(config.model_dump())
        trainer.fit(model_obj, data)
    elif finetune_mode == "full-state":
        # Start a new experiment continuing another run's full training state
        logger.info(f"Continuing the full training state in '{finetune_ckpt}'")
        model_obj = model(config, **model_kwargs)
        tensorboard_logger.log_hyperparams(config.model_dump())
        fit_from_full_state(
            trainer, model_obj, data, finetune_ckpt, config, from_finetune_ckpt=True
        )
    else:
        try:
            model_obj = model.load_from_checkpoint(finetune_ckpt)
        except (TypeError, ValidationError) as e:
            logger.error(f"Unable to load {finetune_ckpt}: {e}")
            sys.exit(1)
        logger.info(f"Model's architecture\n{model_obj}")
        # Check if the trainer has changed (but ignore subdir since it is specific to the run)
        if isinstance(model_obj, StyleTTS2) or isinstance(config, StyleTTS2Config):
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
            raise InvalidConfiguration(textwrap.dedent(f"""
                Sorry, you are a trying to fine-tune a model with a different architecture defined in your configuration than was used during pre-training.

                Please fix your configuration or use a different model.

                Values Changed: {pformat(model_config_diff)}
                """))
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
        # TODO: remove this warning in the release after finetune_mode was added
        if finetune_mode == "weights-only" and not optimizer_config_diff:
            logger.warning(textwrap.dedent("""
                finetune_checkpoint now loads weights only by default; the optimizer, LR schedule and step counter restart at 0.
                To keep them, set training.finetune_mode: full-state.
                To continue a previous run, train in that run's experiment folder (same training.logger settings) with `everyvoice train ... --auto-resume`.
                """))
        # Finetune from Checkpoint
        logger.info(
            f"Fine-tuning from the weights in '{finetune_ckpt}': training logs start from epoch 0/step 0."
        )
        # This will only use the weights in the model_obj, the optimizer and current epoch etc will be restarted using
        # the configuration in model_obj.config, see https://github.com/Lightning-AI/pytorch-lightning/issues/5339
        trainer.fit(model_obj, data)


def _config_diff(ckpt_section, live_section) -> DeepDiff:
    return DeepDiff(
        ckpt_section.model_dump(),
        live_section.model_dump(),
        exclude_types=[Path],
    )


def check_resume_compatible(
    ckpt_config, live_config, ckpt_path, from_finetune_ckpt=False
) -> None:
    """Exit if live_config changes settings a full restore from ckpt_path can't apply.

    A full restore loads the checkpoint's optimizer and scheduler state over the
    optimizers built from live_config, so optimizer changes would be silently
    ignored, and changes to the architecture or to which parameters are trained
    make the restore fail. Other changes, e.g. to the batch size, the data or
    max_steps, do apply, and are logged.
    """
    sections = {
        "training.optimizer": (
            ckpt_config.training.optimizer,
            live_config.training.optimizer,
        ),
        "model": (ckpt_config.model, live_config.model),
    }
    blocking: dict[str, Any] = {
        name: diff
        for name, (ckpt_section, live_section) in sections.items()
        if (diff := _config_diff(ckpt_section, live_section))
    }
    # HiFiGAN's vocoder-matching freezes discriminator layers, which changes
    # the parameters its discriminator optimizer holds
    ckpt_matching = ckpt_config.training.finetune_mode == "vocoder-matching"
    live_matching = live_config.training.finetune_mode == "vocoder-matching"
    if ckpt_matching != live_matching:
        blocking["training.finetune_mode"] = {
            "values_changed": {
                "training.finetune_mode": {
                    "old_value": ckpt_config.training.finetune_mode,
                    "new_value": live_config.training.finetune_mode,
                }
            }
        }
    if blocking:
        changes = "\n".join(
            f"{name}: {pformat(dict(diff))}" for name, diff in blocking.items()
        )
        if from_finetune_ckpt:
            way_out = (
                "set training.finetune_mode: weights-only to load only its weights."
            )
        else:
            way_out = (
                "start a new experiment with "
                f"training.finetune_checkpoint set to '{ckpt_path}' to load only its weights."
            )
        logger.error(
            f"Your configuration changes settings that can't change when training continues "
            f"from the full training state in '{ckpt_path}':\n{changes}\n"
            "The checkpoint's optimizer and learning rate schedule are restored as they are. "
            f"Either revert these settings, or {way_out}"
        )
        sys.exit(1)

    other_changes = DeepDiff(
        ckpt_config.training.model_dump(
            exclude={"optimizer", "logger", "finetune_checkpoint", "finetune_mode"}
        ),
        live_config.training.model_dump(
            exclude={"optimizer", "logger", "finetune_checkpoint", "finetune_mode"}
        ),
        exclude_types=[Path],
    )
    if other_changes:
        logger.info(
            f"Training settings changed since '{ckpt_path}' was saved:\n{other_changes.pretty()}"
        )


def fit_from_full_state(
    trainer, model_obj, data, ckpt_path, config, from_finetune_ckpt=False
) -> None:
    """Fit model_obj continuing the full training state (weights, optimizer, LR
    schedule, step, epoch) in ckpt_path, with the current config.

    from_finetune_ckpt: ckpt_path is finetune_checkpoint (finetune_mode:
    full-state), not this experiment's own last checkpoint.
    """
    import tempfile

    import torch

    from everyvoice.base_cli.slurm import check_training_left

    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=True)
    check_resume_compatible(
        type(config).from_checkpoint_dump(ckpt["hyper_parameters"]["config"]),
        config,
        ckpt_path,
        from_finetune_ckpt=from_finetune_ckpt,
    )
    if from_finetune_ckpt:
        check_training_left(
            ckpt, ckpt_path, config.training.max_steps, config.training.max_epochs
        )
    # We need to create a temporary checkpoint with torch.save because on_save_checkpoint
    # removes all paths for checkpoint portability. However, some paths, like "vocoder_path"
    # should be still accessible when training is resumed.
    ckpt["hyper_parameters"]["config"] = config.model_dump(mode="json")
    # TODO: check if we need to do the same thing with stats and any thing else registered on the model
    # A file in a temporary directory, not a NamedTemporaryFile: Windows can't
    # open a NamedTemporaryFile a second time while it is open
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_ckpt = os.path.join(tmp_dir, "resume.ckpt")
        torch.save(ckpt, tmp_ckpt)
        # This will resume the weights, optimizer, and steps from ckpt_path, see https://github.com/Lightning-AI/pytorch-lightning/issues/5339
        trainer.fit(model_obj, data, ckpt_path=tmp_ckpt)


def inference_base_command(
    model: FastSpeech2,
    config_args: list[str],
):
    config = model.config
    update_config_from_cli_args(config_args, config)
