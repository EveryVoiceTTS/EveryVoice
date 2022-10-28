import json
import os
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import typer
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from smts.config import CONFIGS
from smts.dataloader import FeaturePredictionDataModule, HiFiGANDataModule
from smts.DeepForcedAligner.dfa.dataset import AlignerDataModule
from smts.DeepForcedAligner.dfa.model import Aligner
from smts.DeepForcedAligner.dfa.utils import extract_durations_for_item
from smts.model.feature_prediction.fastspeech2 import FastSpeech2
from smts.model.vocoder.hifigan import HiFiGAN
from smts.preprocessor import Preprocessor
from smts.run_tests import run_tests
from smts.utils import expand_config_string_syntax, update_config

app = typer.Typer(pretty_exceptions_show_locals=False)

_config_keys = {k: k for k in CONFIGS.keys()}

CONFIGS_ENUM = Enum("CONFIGS", _config_keys)  # type: ignore


class PreprocessCategories(str, Enum):
    audio = "audio"
    sox_audio = "sox_audio"
    pitch = "pitch"
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
    aligner = "aligner"
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
    compute_stats: bool = typer.Option(False, "-s", "--stats"),
    overwrite: bool = typer.Option(False, "-O", "--overwrite"),
):
    config = CONFIGS[name.value]
    preprocessor = Preprocessor(config)
    to_preprocess = {k: k in data for k in PreprocessCategories.__members__.keys()}  # type: ignore
    if not data:
        logger.info(
            f"No specific preprocessing data requested, processing everything (pitch, mel, energy, durations, inputs) from dataset '{name}'"
        )
    else:
        preprocessor.preprocess(
            output_path=output_path,
            process_audio=to_preprocess["audio"],
            process_sox_audio=to_preprocess["sox_audio"],
            process_spec=to_preprocess["mel"],
            process_energy=to_preprocess["energy"],
            process_pitch=to_preprocess["pitch"],
            process_duration=to_preprocess["dur"],
            process_pfs=to_preprocess["feats"],
            process_text=to_preprocess["text"],
            overwrite=overwrite,
        )
    if compute_stats:
        preprocessor.compute_stats(overwrite=overwrite)


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
def extract_alignments(
    name: CONFIGS_ENUM,
    accelerator: str = typer.Option("auto"),
    devices: str = typer.Option("auto"),
    model_path: Path = typer.Option(
        default=None, exists=True, file_okay=True, dir_okay=False
    ),
    config: List[str] = typer.Option(None),
    config_path: Path = typer.Option(None, exists=True, dir_okay=False, file_okay=True),
    num_processes: int = typer.Option(None),
):
    # TODO: make this faster
    if num_processes is None:
        num_processes = 4
    original_config = CONFIGS[name.value]
    if config is not None and config:
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
    data = AlignerDataModule(config)
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
    )
    if model_path:
        model = Aligner.load_from_checkpoint(model_path)
        # TODO: check into the best way to update config from re-loaded model
        # model.update_config(config)
        model.config = config
        trainer.predict(model, dataloaders=data)
    else:
        trainer.predict(dataloaders=data)
    sep = config["preprocessing"]["value_separator"]
    save_dir = Path(config["preprocessing"]["save_dir"])
    for item in tqdm(
        data.predict_dataloader().dataset,
        total=len(data.predict_dataloader().dataset),
    ):
        basename = item["basename"]
        speaker = item["speaker"]
        language = item["language"]
        tokens = item["tokens"].cpu()
        pred = np.load(
            save_dir / sep.join([basename, speaker, language, "duration.npy"])
        )
        item, durations = extract_durations_for_item(
            item,
            tokens,
            pred,
            method=config["training"]["aligner"]["extraction_method"],
        )
        torch.save(
            torch.tensor(durations).long(),
            save_dir / sep.join([basename, speaker, language, "duration.npy"]),
        )


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
    if config is not None and config:
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
    tensorboard_logger = TensorBoardLogger(**config["training"]["logger"])
    lr_monitor = LearningRateMonitor(logging_interval="step")

    if model.value == "aligner":
        logger.info("Starting training for alignment model.")
        ckpt_callback = ModelCheckpoint(
            monitor="validation/loss",
            mode="min",
            save_last=True,
            save_top_k=config["training"]["aligner"]["save_top_k_ckpts"],
            every_n_train_steps=config["training"]["aligner"]["ckpt_steps"],
            every_n_epochs=config["training"]["aligner"]["ckpt_epochs"],
        )
        trainer = Trainer(
            gradient_clip_val=1.0,
            logger=tensorboard_logger,
            accelerator=accelerator,
            devices=devices,
            max_epochs=config["training"]["aligner"]["max_epochs"],
            callbacks=[ckpt_callback, lr_monitor],
            strategy=strategy,
            detect_anomaly=False,  # used for debugging, but triples training time
        )
        aligner = Aligner(config)
        data = AlignerDataModule(config)
        last_ckpt = (
            config["training"]["aligner"]["finetune_checkpoint"]
            if os.path.exists(config["training"]["aligner"]["finetune_checkpoint"])
            else None
        )
        tensorboard_logger.log_hyperparams(config)
        trainer.fit(aligner, data, ckpt_path=last_ckpt)

    if model.value == "hifigan":
        logger.info("Starting training for HiFiGAN model.")
        ckpt_callback = ModelCheckpoint(
            monitor="validation/mel_spec_error",
            mode="min",
            save_last=True,
            save_top_k=config["training"]["vocoder"]["save_top_k_ckpts"],
            every_n_train_steps=config["training"]["vocoder"]["ckpt_steps"],
            every_n_epochs=config["training"]["vocoder"]["ckpt_epochs"],
        )
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
        ckpt_callback = ModelCheckpoint(
            monitor="eval/total_loss",
            mode="min",
            save_last=True,
            save_top_k=config["training"]["feature_prediction"]["save_top_k_ckpts"],
            every_n_train_steps=config["training"]["feature_prediction"]["ckpt_steps"],
            every_n_epochs=config["training"]["feature_prediction"]["ckpt_epochs"],
        )
        trainer = Trainer(
            logger=tensorboard_logger,
            accelerator=accelerator,
            devices=devices,
            max_epochs=config["training"]["feature_prediction"]["max_epochs"],
            callbacks=[ckpt_callback, lr_monitor],
            strategy=strategy,
            detect_anomaly=False,  # used for debugging, but triples training time
            fast_dev_run=True,  # used for debugging, touches every piece of code
            overfit_batches=0.0,  # used for debugging, overfits to <int> batches or <float> % of training data
        )
        feature_prediction_network = FastSpeech2(config)

        data = FeaturePredictionDataModule(config)
        last_ckpt = (
            config["training"]["feature_prediction"]["finetune_checkpoint"]
            if os.path.exists(
                config["training"]["feature_prediction"]["finetune_checkpoint"]
            )
            else None
        )
        tensorboard_logger.log_hyperparams(config)
        trainer.fit(feature_prediction_network, data, ckpt_path=last_ckpt)
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
