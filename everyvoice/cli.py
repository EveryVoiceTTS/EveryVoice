import json
from enum import Enum
from pathlib import Path
from typing import List

import questionary
import typer
from loguru import logger
from rich import print
from rich.panel import Panel
from rich.prompt import Prompt
from rich.style import Style
from slugify import slugify

from everyvoice.config import CONFIGS
from everyvoice.model.aligner.DeepForcedAligner.dfaligner.cli import app as dfaligner_app
from everyvoice.model.e2e.cli import app as e2e_app
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli import app as fs2_app
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import app as hfgl_app

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


def questionary_callback(value: str, name: str):
    path = Path(value)
    if path.is_file():
        return f"Sorry, the path at '{path.absolute()}' is a file. Please select a directory."
    path = path / slugify(name)
    if path.exists():
        return f"Sorry, the path at '{path.absolute()}' already exists. Please choose another output directory."
    return True


def questionary_file(value: str):
    path = Path(value)
    if path.is_dir():
        return f"Sorry, the path at '{path.absolute()}' is a directory. Please select a file."
    return True


def questionary_dir(value: str):
    path = Path(value)
    if path.is_file():
        return f"Sorry, the path at '{path.absolute()}' is a file. Please select a directory."
    return True


@app.command(
    help="""This command will help you create all the configuration necessary for your using your dataset.

In order to get started, please:
 - have your audio data (in .wav format) together in a folder
 - have a 'metadata' file that minimally has two columns, one for the text of the audio and one for the basename of the file

Inside /path/to/wavs/ you should have wav audio files like test0001.wav, test0002.wav etc - they can be called anything you want. but the part of the file (minus the .wav portion) must be in your metadata file.

Example wavs directory: "/path/to/wavs/"

Example metadata: https://github.com/roedoejet/EveryVoice/blob/main/everyvoice/filelists/lj_full.psv

"""
)
def config_wizard(
    name: str = typer.Option(
        ...,
        prompt="What would you like to call this dataset?",
        help="The name of your dataset.",
    ),
):
    output_dir = Path(
        questionary.path(
            "Where should the wizard save your files?",
            default=".",
            validate=lambda x: questionary_callback(x, name),
        ).ask()
    )
    wavs_dir = Path(
        questionary.path("Where are your audio files?", validate=questionary_dir).ask()
    )
    filelist_path = Path(
        questionary.path(
            "Where is your data filelist?", validate=questionary_file
        ).ask()
    )
    from everyvoice.config.preprocessing_config import (
        AudioConfig,
        Dataset,
        PreprocessingConfig,
    )
    from everyvoice.config.shared_types import BaseTrainingConfig, LoggerConfig
    from everyvoice.config.text_config import Symbols, TextConfig
    from everyvoice.model.aligner.config import AlignerConfig
    from everyvoice.model.e2e.config import EveryVoiceConfig
    from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
    from everyvoice.model.vocoder.config import VocoderConfig
    from everyvoice.utils import generic_csv_loader, read_festival, write_dict
    from everyvoice.utils.cli_wizard import (
        auto_check_audio,
        create_default_filelist,
        create_sox_effects_list,
        get_lang_information,
        get_menu_prompt,
        get_required_headers,
        get_single_lang_information,
        get_symbol_set,
        write_config_to_file,
        write_dict_to_config,
    )

    logger.info(
        f"Great! Launching Configuration Wizard 🧙 for dataset named {name} with audio files at {wavs_dir} and a filelist at {filelist_path}. Files will be output here: {output_dir.absolute()}"
    )
    output_path = output_dir / name
    output_path.mkdir(parents=True, exist_ok=True)
    # Determine file type
    file_type_choices = ["psv", "tsv", "csv", "festival"]
    file_type_choice = get_menu_prompt(
        "Select which format your filelist is in:",
        choices=file_type_choices,
        multi=False,
        search=False,
    )
    file_type = file_type_choices[file_type_choice]
    if file_type == "csv":
        delimiter = ","
    elif file_type == "psv":
        delimiter = "|"
    elif file_type == "tsv":
        delimiter = "\t"
    filelist_data_dict = None
    if file_type == "festival":
        filelist_data_dict = read_festival(filelist_path)
        headers = list(filelist_data_dict[0].keys())
        filelist_data = [list(row.values()) for row in filelist_data_dict]
    else:
        # Check filelist headers
        filelist_data = generic_csv_loader(filelist_path, delimiter=delimiter)
        headers = list(filelist_data[0])
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
    langs_inferred_from_data = None
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
            langs_inferred_from_data = {x[int(lang_column)] for x in filelist_data}
    # TODO: test with multilingual and monolingual labelled (with recognized and unrecognized) and unlabelled datasets
    if langs_inferred_from_data is None:
        # No language selected
        ds_supported_langs, ds_unsupported_langs = get_lang_information()
        number_of_languages = len(ds_supported_langs) + len(ds_unsupported_langs)
        if number_of_languages > 1:
            logger.info(
                "Sorry, if your dataset has more than one language in it, you will have to add this information to your filelist, because the wizard can't guess!"
            )
    else:
        langs_to_convert = (
            {}
        )  # we might need to convert some of the names of languages to the recognized iso code
        logger.info("Getting supported languages...")
        from pycountry import languages
        from readalongs.util import get_langs

        supported_langs = get_langs()[1]
        all_langs = list(languages)
        ds_supported_langs = {}
        ds_unsupported_langs = {}
        for lang in langs_inferred_from_data:
            if lang in supported_langs and typer.confirm(
                f"Is the lang in your data labelled {lang} the same as the language {supported_langs[lang]} with the iso code {lang}?"
            ):
                ds_supported_langs[lang] = supported_langs[lang]
                continue
            logger.info(
                f"We couldn't recognize {lang}. Please choose it from our list:"
            )
            new_lang_sup, new_lang_unsup = get_single_lang_information(
                supported_langs.items(), all_langs
            )
            if new_lang_sup:
                ds_supported_langs = {**ds_supported_langs, **new_lang_sup}
            if new_lang_unsup:
                ds_unsupported_langs = {**ds_unsupported_langs, **new_lang_unsup}
            langs_to_convert[lang] = (
                supported_langs.keys()[new_lang_sup]
                if new_lang_sup
                else all_langs[new_lang_unsup].alpha_3  # type: ignore
            )
        if langs_to_convert:
            # replace all unrecognized languages with recognized ones
            for k, v in langs_to_convert.items():
                for i, row in enumerate(filelist_data):
                    if row[lang_column] == k:
                        filelist_data[i][lang_column] = v
    filelist_data = create_default_filelist(filelist_data, headers)
    symbol_set = Symbols(
        **get_symbol_set(filelist_data, ds_supported_langs, ds_unsupported_langs)
    )
    dataset_filelist = output_path / "filelist.psv"
    write_dict(dataset_filelist, filelist_data, headers)
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
    preprocessed_filelist_path = output_path / "preprocessed" / "processed_filelist.psv"
    audio_config = AudioConfig(
        min_audio_length=min_length,
        max_audio_length=max_length,
        input_sampling_rate=input_sr,
        output_sampling_rate=input_sr,
        alignment_sampling_rate=input_sr,
    )
    dataset_config = Dataset(
        label=name,
        data_dir=wavs_dir.absolute(),
        filelist=dataset_filelist.absolute(),
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
    text_config = TextConfig(symbols=symbol_set)
    text_config_path = (config_dir / f"text.{config_format}").absolute()
    write_config_to_file(text_config, text_config_path)
    ## Create Aligner Config
    aligner_logger = LoggerConfig(name="AlignerExperiment", save_dir=log_dir)
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
    fp_logger = LoggerConfig(name="FeaturePredictionExperiment", save_dir=log_dir)
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
    vocoder_logger = LoggerConfig(name="VocoderExperiment", save_dir=log_dir)
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
    e2e_logger = LoggerConfig(name="E2E-Experiment", save_dir=log_dir)
    e2e_config = EveryVoiceConfig(
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
    print(
        Panel(
            f"You've finished configuring your dataset. Your files are located at {config_dir.absolute()}",
            title="Congratulations 🎉",
            subtitle="Next Steps Documentation: http://localhost:8000/guides/index.html",
            border_style=Style(color="#0B4F19"),
        )
    )


@app.command()
def test(suite: TestSuites = typer.Argument(TestSuites.dev)):
    """This command will run the test suite specified by the user"""
    from everyvoice.run_tests import run_tests

    run_tests(suite)


CLICK_APP = typer.main.get_group(app)

if __name__ == "__main__":
    app()
