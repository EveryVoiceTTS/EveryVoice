import json
import os
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from smts.config import CONFIGS
from smts.dataloader import HiFiGANDataModule
from smts.model.vocoder.hifigan import HiFiGAN
from smts.preprocessor import Preprocessor
from smts.run_tests import run_tests
from smts.utils import expand_config_string_syntax, update_config

app = typer.Typer()

_config_keys = {k: k for k in CONFIGS.keys()}

CONFIGS_ENUM = Enum("CONFIGS", _config_keys)  # type: ignore


class PreprocessCategories(str, Enum):
    audio = "audio"
    sox_audio = "sox_audio"
    f0 = "f0"
    mel = "mel"
    energy = "energy"
    dur = "dur"
    text = "text"
    feats = "feats"


class TestSuites(str, Enum):
    all = "all"
    configs = "configs"
    dev = "dev"
    model = "model"
    preprocessing = "preprocessing"
    text = "text"


class Model(str, Enum):
    hifigan = "hifigan"
    feat = "feat"


@app.command()
def test(suite: TestSuites = typer.Argument(TestSuites.dev)):
    """This command will run the test suite specified by the user"""
    run_tests(suite)


@app.command()
def preprocess(
    name: CONFIGS_ENUM,
    data: Optional[List[PreprocessCategories]] = typer.Option(None, "-d", "--data"),
    output_path: Optional[Path] = typer.Option(
        "processed_filelist.psv", "-o", "--output"
    ),
    overwrite: bool = typer.Option(False, "-O", "--overwrite"),
):
    config = CONFIGS[name.value]
    preprocessor = Preprocessor(config)
    to_preprocess = {k: k in data for k in PreprocessCategories.__members__.keys()}  # type: ignore
    if not data:
        logger.info(
            f"No specific preprocessing data requested, processing everything (f0, mel, energy, durations, inputs) from dataset '{name}'"
        )
    else:
        preprocessor.preprocess(
            output_path=output_path,
            process_audio=to_preprocess["audio"],
            process_sox_audio=to_preprocess["sox_audio"],
            process_spec=to_preprocess["mel"],
            process_energy=to_preprocess["energy"],
            process_f0=to_preprocess["f0"],
            process_duration=to_preprocess["dur"],
            process_pfs=to_preprocess["feats"],
            process_text=to_preprocess["text"],
            overwrite=overwrite,
        )


@app.command()
def synthesize(
    text: str,
    model_path: Path = typer.Option(
        default=None, exists=True, file_okay=True, dir_okay=False
    ),
):
    # TODO: allow for inference parameters like speaker, language etc
    logger.info(f"Synthesizing {text} from model at {model_path}.")


@app.command()
def train(
    name: CONFIGS_ENUM,
    accelerator: str = typer.Option("auto"),
    devices: str = typer.Option("auto"),
    model: Model = typer.Option(Model.hifigan),
    strategy: str = typer.Option(None),
    config: List[str] = typer.Option(None),
    config_path: Path = typer.Option(None, exists=True, dir_okay=False, file_okay=True),
):
    original_config = CONFIGS[name.value]
    if config is not None:
        for update in config:
            key, value = update.split("=")
            logger.info(f"Updating config '{key}' to value '{value}'")
            original_config = update_config(
                original_config, expand_config_string_syntax(update)
            )
    else:
        config = original_config
    if config_path is not None:
        logger.info(f"Loading and updating config from '{config_path}'")
        config_override = json.load(config_path)
        config = update_config(config, config_override)
    if model.value == "hifigan":
        tensorboard_logger = TensorBoardLogger(**config["training"]["logger"])
        logger.info("Starting training for HiFiGAN model.")
        ckpt_callback = ModelCheckpoint(
            monitor="validation/mel_spec_error",
            mode="min",
            save_last=True,
            save_top_k=config["training"]["vocoder"]["save_top_k_ckpts"],
            every_n_train_steps=config["training"]["vocoder"]["ckpt_steps"],
            every_n_epochs=config["training"]["vocoder"]["ckpt_epochs"],
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        trainer = Trainer(
            logger=tensorboard_logger,
            accelerator=accelerator,
            devices=devices,
            max_epochs=config["training"]["vocoder"]["max_epochs"],
            callbacks=[ckpt_callback, lr_monitor],
            strategy=strategy,
            detect_anomaly=False,  # used for debugging, but triples training time
        )
        vocoder = HiFiGAN(config)
        data = HiFiGANDataModule(config)
        last_ckpt = (
            config["training"]["vocoder"]["finetune_checkpoint"]
            if os.path.exists(config["training"]["vocoder"]["finetune_checkpoint"])
            else None
        )
        tensorboard_logger.log_hyperparams(config)
        trainer.fit(vocoder, data, ckpt_path=last_ckpt)
    # TODO: allow for updating hyperparameters from CLI

    if model.value == "feat":
        logger.info("Starting training for feature prediction model.")
        # TODO: implement feature prediction model tuning


@app.command()
def tune(
    name: CONFIGS_ENUM,
    accelerator: str = typer.Option("auto"),
    devices: str = typer.Option("auto"),
    model: Model = typer.Option(Model.hifigan),
):
    config = CONFIGS[name.value]

    if model.value == "hifigan":
        logger.info("Starting hyperparameter tuning for HiFiGAN model.")
        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=config["training"]["vocoder"]["max_epochs"],
            auto_scale_batch_size=True,
        )
        vocoder = HiFiGAN(config)
        data = HiFiGANDataModule(config)
        trainer.tune(vocoder, data)

    if model.value == "feat":
        logger.info("Starting hyperparameter tuning for feature prediction model.")
        # TODO: implement feature prediction model tuning


if __name__ == "__main__":
    app()
