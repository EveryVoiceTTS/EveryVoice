import json
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger
from rich.prompt import Prompt
from slugify import slugify

from smts.config import CONFIGS
from smts.model.aligner.DeepForcedAligner.dfaligner.cli import app as dfaligner_app
from smts.model.e2e.cli import app as e2e_app
from smts.model.e2e.config import SMTSConfig  # type: ignore
from smts.model.feature_prediction.FastSpeech2_lightning.fs2.cli import app as fs2_app
from smts.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import app as hfgl_app

app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(dfaligner_app, name="dfa")
app.add_typer(e2e_app, name="e2e")
app.add_typer(hfgl_app, name="hifigan")
app.add_typer(fs2_app, name="fs2")

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
    e2e = "e2e"
    hifigan = "hifigan"
    feat = "feat"


def output_callback(ctx: typer.Context, value: Path):
    path = value / slugify(ctx.params["name"])
    if path.exists():
        raise typer.BadParameter(
            f"Sorry, the path at '{path.absolute()}' already exists. Please choose another output directory."
        )
    return value


@app.command()
def config_wizard(
    name: str = typer.Option(
        "test", prompt="What would you like to call this dataset?"
    ),
    output_dir: Path = typer.Option(
        ".",
        dir_okay=True,
        file_okay=False,
        prompt="Where should the wizard save your files?",
        callback=output_callback,
    ),
    wavs_dir: Path = typer.Option(
        "/home/aip000/tts/code/SmallTeamSpeech/smts/data/lj/wavs",
        exists=True,
        dir_okay=True,
        file_okay=False,
        prompt="What is the absolute path to your audio files?",
    ),
    filelist_path: Path = typer.Option(
        "/home/aip000/tts/code/SmallTeamSpeech/smts/filelists/lj_test.psv",
        exists=True,
        dir_okay=False,
        file_okay=True,
        prompt="What is the absolute path to data filelist?",
    ),
):
    from smts.config.preprocessing_config import (
        AudioConfig,
        Dataset,
        PreprocessingConfig,
    )
    from smts.config.shared_types import BaseTrainingConfig, LoggerConfig
    from smts.config.text_config import TextConfig
    from smts.model.aligner.config import AlignerConfig
    from smts.model.e2e.config import SMTSConfig
    from smts.model.feature_prediction.config import FeaturePredictionConfig
    from smts.model.vocoder.config import VocoderConfig
    from smts.utils import generic_csv_loader, write_dict
    from smts.utils.cli_wizard import (
        auto_check_audio,
        create_default_filelist,
        create_sox_effects_list,
        get_menu_prompt,
        get_required_headers,
        write_config_to_file,
        write_dict_to_config,
    )

    logger.info(
        f"Great! Launching Configuration Wizard 🧙 for dataset named {name} with audio files at {wavs_dir} and a filelist at {filelist_path}. Files will be output here: {output_dir.absolute()}"
    )
    output_path = output_dir / name
    output_path.mkdir(parents=True, exist_ok=True)
    # Check filelist headers
    filelist_data = generic_csv_loader(filelist_path)
    headers = [x for x in filelist_data[0]]
    if "basename" not in headers or "text" not in headers:
        logger.info(
            "Your filelist must minimally contain a 'basename' and 'text' column, but yours does not."
        )
        headers = get_required_headers(headers, sample_data=filelist_data[:5])
    for i, h in enumerate(headers):
        if h not in ["text", "basename", "speaker", "language"]:
            headers[i] = f"unknown_{i}"
    if "speaker" not in headers:
        spkr_column = Prompt.ask(
            "Do you have a column for speaker id?",
            choices=["no"]
            + [
                str(x)
                for x in range(len(headers))
                if headers[x] not in ["text", "basename"]
            ],
            default="no",
        )
        if spkr_column != "no":
            headers[int(spkr_column)] = "speaker"
        else:
            headers.append("speaker")
    if "language" not in headers:
        lang_column = Prompt.ask(
            "Do you have a column for language id?",
            choices=["no"]
            + [
                str(x)
                for x in range(len(headers))
                if headers[x] not in ["text", "basename", "speaker"]
            ],
            default="no",
        )
        if lang_column != "no":
            headers[int(lang_column)] = "language"
        else:
            headers.append("language")
    filelist_data = create_default_filelist(filelist_data, headers)
    write_dict(output_path / "filelist.psv", filelist_data, headers)
    # Check Audio
    auto_check = Prompt.ask(
        "Config Wizard can try to determine the appropriate settings for your audio automatically. Would you like to do that or manually specify information (like sample rates etc)?",
        choices=["auto", "manual"],
        default="auto",
    )
    if auto_check == "auto":
        input_sr, min_length, max_length = auto_check_audio(filelist_data, wavs_dir)
    else:
        input_sr = int(
            Prompt.ask(
                "What is the input sampling rate of your data in Hertz?",
                default="22050",
            )
        )
        min_length = 0.25
        max_length = 11.0
        logger.info(
            "Note: the configuration wizard does not currently support super resolution. If you would like to synthesize to a different sampling rate than your input, please visit the docs: https://pathtodocs"
        )
    sox_effects = ["none", "resample", "norm", "sil-start"]
    sox_indices: List[int] = get_menu_prompt(  # type: ignore
        "Please select zero or more from the following audio preprocessing options:",
        [
            "none",
            "resample_audio (will resample to previously defined sampling rate)",
            "normalize to -3.0dB",
            "remove silence at the beginning of each clip",
        ],
        multi=True,
    )
    sox_config = create_sox_effects_list(
        [sox_effects[i] for i in sox_indices], input_sr
    )
    config_formats = ["yaml", "json"]
    config_format_index: int = get_menu_prompt(  # type: ignore
        "What kind of configuration would you like to generate?", config_formats
    )
    config_format = config_formats[config_format_index]
    # create_config_files
    config_dir = output_path / "config"
    config_dir.mkdir(exist_ok=True, parents=True)
    # log dir
    log_dir = (output_path / "logs").absolute()
    log_dir.mkdir(parents=True, exist_ok=True)
    ## Create Preprocessing Config
    preprocessed_filelist_path = (
        output_path / "preprocessed" / "preprocessed_filelist.psv"
    )
    audio_config = AudioConfig(
        min_audio_length=min_length,
        max_audio_length=max_length,
        input_sampling_rate=input_sr,
        output_sampling_rate=input_sr,
    )
    dataset_config = Dataset(
        label=name,
        data_dir=wavs_dir.absolute(),
        filelist=filelist_path.absolute(),
        sox_effects=sox_config,
    )
    preprocessing_config = PreprocessingConfig(
        dataset=name,
        save_dir=(output_path / "preprocessed").absolute(),
        audio=audio_config,
        source_data=[dataset_config],
    )
    preprocessing_config_path = (
        config_dir / f"preprocessing.{config_format}"
    ).absolute()
    write_config_to_file(preprocessing_config, preprocessing_config_path)
    ## Create Text Config
    text_config = TextConfig()
    text_config_path = (config_dir / f"text.{config_format}").absolute()
    write_config_to_file(text_config, text_config_path)
    ## Create Aligner Config
    aligner_logger = LoggerConfig(name="Aligner Experiment", save_dir=log_dir)
    aligner_config = AlignerConfig(
        training=BaseTrainingConfig(
            filelist=preprocessed_filelist_path.absolute(), logger=aligner_logger
        )
    )
    aligner_config_path = (config_dir / f"aligner.{config_format}").absolute()
    aligner_config_json = json.loads(aligner_config.json())
    aligner_config_json["preprocessing"] = str(preprocessing_config_path)
    aligner_config_json["text"] = str(text_config_path)
    write_dict_to_config(aligner_config_json, aligner_config_path)
    ## Create Feature Prediction Config
    fp_logger = LoggerConfig(name="Feature Prediction Experiment", save_dir=log_dir)
    fp_config = FeaturePredictionConfig(
        training=BaseTrainingConfig(
            filelist=preprocessed_filelist_path.absolute(), logger=fp_logger
        )
    )
    fp_config_path = (config_dir / f"feature_prediction.{config_format}").absolute()
    fp_config_json = json.loads(fp_config.json())
    fp_config_json["preprocessing"] = str(preprocessing_config_path)
    fp_config_json["text"] = str(text_config_path)
    write_dict_to_config(fp_config_json, fp_config_path)
    ## Create Vocoder Config
    vocoder_logger = LoggerConfig(name="Vocoder Experiment", save_dir=log_dir)
    vocoder_config = VocoderConfig(
        training=BaseTrainingConfig(
            filelist=preprocessed_filelist_path.absolute(), logger=vocoder_logger
        )
    )
    vocoder_config_path = (config_dir / f"vocoder.{config_format}").absolute()
    vocoder_config_json = json.loads(vocoder_config.json())
    vocoder_config_json["preprocessing"] = str(preprocessing_config_path)
    write_dict_to_config(vocoder_config_json, vocoder_config_path)
    ## E2E Config
    e2e_logger = LoggerConfig(name="E2E Experiment", save_dir=log_dir)
    e2e_config = SMTSConfig(
        vocoder=vocoder_config_path,
        feature_prediction=fp_config_path,
        aligner=aligner_config_path,
        training=BaseTrainingConfig(
            filelist=preprocessed_filelist_path.absolute(), logger=e2e_logger
        ),
    )
    e2e_config_json = json.loads(e2e_config.json())
    e2e_config_json["aligner"] = str(aligner_config_path)
    e2e_config_json["feature_prediction"] = str(fp_config_path)
    e2e_config_json["vocoder"] = str(vocoder_config_path)
    e2e_config_path = (config_dir / f"e2e.{config_format}").absolute()
    write_dict_to_config(e2e_config_json, e2e_config_path)
    # TODO: print next steps
    print("TODO: print next steps")


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
    # TODO: which preprocessing config to use?
    preprocessor = Preprocessor(config)  # type: ignore
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
            compute_stats=compute_stats,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    app()
