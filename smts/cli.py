from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger

from smts.config import CONFIGS
from smts.config.base_config import SMTSConfig  # type: ignore
from smts.model.aligner.DeepForcedAligner.dfaligner.cli import app as dfaligner_app
from smts.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import app as hfgl_app

app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(dfaligner_app, name="dfa")
app.add_typer(hfgl_app, name="hifigan")

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
    from smts.run_tests import run_tests

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
    from smts.preprocessor import Preprocessor

    config: SMTSConfig = SMTSConfig.load_config_from_path(CONFIGS[name.value])
    preprocessor = Preprocessor(config)  # TODO: which preprocessing config to use?
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


if __name__ == "__main__":
    app()
