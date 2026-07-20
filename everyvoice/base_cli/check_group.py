"""
CLI command to check EveryVoice data and/or configs
"""

from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.check_data import (
    check_data_command,
)

from . import command, default_typer_args
from .checkpoint import load_checkpoint
from .interfaces import typer_file_option

# check group
check_group = typer.Typer(**default_typer_args)

# Add check_data to check_group
command(
    check_group,
    name="data",
    short_help="Check your data for outliers or any anomalies",
    help="""
    # Check Data Help

    This command will check all of your data to help you find anomalies and outliers.

    To check your data, make sure you've run preprocessing first (everyvoice preprocess --help).
    Then you need to briefly and partially train a text-to-spec model. We recommend 100-1000 steps to start.

    Then, with your partially trained model you can run the data checker:
    \n\n
    **everyvoice check-data config/everyvoice-text-to-spec.yaml logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt**
    \n\n

    This will output two files - one containing some basic statistics for your data and the other containing losses for each datapoint as calculated by your model.

    """,
)(check_data_command)


def require_exactly_one_of(arg1: Any, arg1_name: str, arg2: Any, arg2_name: str):
    if arg1 and arg2:
        raise typer.BadParameter(
            f"Please specify only one of {arg1_name} or {arg2_name}."
        )
    if not arg1 and not arg2:
        raise typer.BadParameter(f"One of {arg1_name} and {arg2_name} is required.")


def open_text_or_psv_file(
    text_file: Optional[Path], psv_file: Optional[Path], language: Optional[str]
) -> list[dict[str, str]]:
    """helper for check_text_config: Open a text or psv file into records.

    Language is required if not already in the psv

    raises: typer.BadParameter if something is wrong"""
    from everyvoice.utils import generic_psv_filelist_reader

    if text_file:
        with open(text_file, "r", encoding="utf8") as f:
            text_lines = list(f)
        # print(text_lines)
        if language is None:
            raise typer.BadParameter("--language is required with --text-file.")
        records = [{"characters": line, "language": language} for line in text_lines]
    else:
        assert psv_file
        records = generic_psv_filelist_reader(psv_file)
        if "language" not in records[0]:
            if language is None:
                raise typer.BadParameter(
                    "--language is required for a psv file without a language column."
                )
            for record in records:
                record["language"] = language
    return records


def get_text_config_from_config_or_model(config: Optional[Path], model: Optional[Path]):
    """Helper for check_text_config: load a TextConfig from a config file or model file"""
    from everyvoice.config.text_config import TextConfig
    from everyvoice.utils import spinner

    if config:
        text_config: TextConfig = TextConfig.load_config_from_path(config)
    else:
        assert model
        with spinner("Loading model"):
            try:
                checkpoint = load_checkpoint(model)
            except Exception as e:
                raise typer.BadParameter(
                    f"Model/checkpoint '{model}' does not appear to be valid.\nError from loader: {e}"
                )
        # print("Looking for text config")
        model_config = checkpoint["hyper_parameters"]["config"]
        if "text" in model_config:
            # FS2 models have hyper_parameters.config.text
            text_config = TextConfig(**model_config["text"])
        elif "ev_config" in model_config and "text" in model_config["ev_config"]:
            # StyleTTS2 models have hyper_parameters.config.ev_config.text
            text_config = TextConfig(**model_config["ev_config"]["text"])
        else:
            # Models without text config, e.g., a HiFiGan Vocoder, are not accepted here
            raise typer.BadParameter(
                f"Model/checkpoint '{model}' does not have an embedded text configuration."
            )
    return text_config


@command(
    check_group,
    name="text-config",
    short_help="Inspect a text configuration for compatiblity with an input file",
)
def check_text_config(
    config: Annotated[
        Optional[Path],
        typer_file_option(
            "--config",
            "-c",
            help="path to text config, i.e., everyvoice-shared-text.yaml",
        ),
    ] = None,
    model: Annotated[
        Optional[Path],
        typer_file_option(
            "--model", "-m", help="path to a model whose text config will be used"
        ),
    ] = None,
    text_file: Annotated[
        Optional[Path],
        typer_file_option(help="path to a plain text file to check"),
    ] = None,
    psv_file: Annotated[
        Optional[Path],
        typer_file_option(help="path to a psv file to check"),
    ] = None,
    language: Annotated[
        Optional[str],
        typer.Option(
            "--language",
            "-l",
            help="language id, required with --text-file, or for a psv file without a language column. "
            + "Declaring the language is always required, because text normalization can be language specific, and g2p is always language specific.",
        ),
    ] = None,
):
    """
    # Check Text Config Help

    Inspect a text configuration for compatiblity with an input file

    Test processing input_file against the text configuration provided, or the text
    configuration found in model, and report any incompatibilities.

    Required options: one of --config and --model, as well as one of --text-file and --psv-file.
    """
    from everyvoice.utils import spinner

    require_exactly_one_of(config, "--config", model, "--model")
    require_exactly_one_of(text_file, "--text-file", psv_file, "--psv-file")
    records = open_text_or_psv_file(text_file, psv_file, language)

    # Expensive imports are deferred so we fail fast where we can
    with spinner("Loading software"):
        from everyvoice.config.text_config import TextConfig  # noqa F401
        from everyvoice.preprocessor.preprocessor import Preprocessor
        from everyvoice.text.text_processor import TextProcessor
        from everyvoice.text.utils import guess_graphemes_in_text

    text_config = get_text_config_from_config_or_model(config, model)
    # print(text_config)

    text_processor_chars_only = TextProcessor(text_config)
    text_processor_all = TextProcessor(text_config)
    with spinner("Analyzing text"):
        for record in records:
            # print(record)
            # Process just the text to calculate missing characters
            _ = Preprocessor.process_text(
                record,
                text_processor_chars_only,
                specific_text_representation=TargetTrainingTextRepresentationLevel.characters,
            )
            # Process all to also calculate missing phones
            _ = Preprocessor.process_text(record, text_processor_all)

    missing_characters = text_processor_chars_only.missing_symbols
    missing_phones = text_processor_all.missing_symbols - missing_characters
    missing_symbol_groups = list(missing_characters)
    for missing_symbol_group in missing_symbol_groups:
        split_symbols = guess_graphemes_in_text(missing_symbol_group)
        if len(split_symbols) > 1:
            count = missing_characters.pop(missing_symbol_group)
            for symbol in split_symbols:
                missing_characters[symbol] += count
    # print("Missing characters", missing_characters)
    # print("Missing phones", missing_phones)
    if missing_characters:
        print(
            "The following characters are missing from your text config:",
            sorted(missing_characters),
        )
    if missing_phones:
        print(
            "The following phones are missing from your text config:",
            sorted(missing_phones),
        )
