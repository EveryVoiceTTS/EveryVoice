from enum import Enum
from pathlib import Path
from pprint import pprint
from typing import List, Optional

import typer
from loguru import logger

from SmallTeamSpeech.config import CONFIGS

app = typer.Typer()


class PreprocessCategories(str, Enum):
    f0 = "f0"
    mel = "mel"
    energy = "energy"
    dur = "dur"
    feats = "feats"


@app.command()
def config(name: str):
    if name not in CONFIGS:
        logger.error(
            f"Sorry, the configuration '{name}' hasn't been defined yet. Please define it."
        )
    logger.info(f"Below is the configuration for '{name}':")
    pprint(CONFIGS[name])


@app.command()
def preprocess(
    name: str, data: Optional[List[PreprocessCategories]] = typer.Option(None)
):
    if not data:
        logger.info(
            f"No specific preprocessing data requested, processing everything (f0, mel, energy, durations, inputs) from dataset {name}"
        )
    else:
        for d in data:
            logger.info(f"Processing {d} from dataset {name}")


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
def train(name: str):
    # TODO: allow for updating hyperparameters from CLI
    # TODO: allow for fine-tuning or continuing from checkpoint
    logger.info(f"Starting training for {name} model.")


if __name__ == "__main__":
    app()
