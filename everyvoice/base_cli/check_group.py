"""
CLI command to check EveryVoice data and/or configs
"""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Optional

import typer

from everyvoice.config.type_definitions import (
    DatasetTextRepresentation,
    TargetTrainingTextRepresentationLevel,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.check_data import (
    check_data_command,
)

from . import command, default_typer_args
from .checkpoint import load_checkpoint
from .interfaces import typer_file_option

if TYPE_CHECKING:
    from everyvoice.config.text_config import TextConfig

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
        text_columns = {r.value for r in DatasetTextRepresentation}
        if not text_columns & records[0].keys():
            raise typer.BadParameter(
                f"'{psv_file}' has none of the columns {sorted(text_columns)} so there is no "
                f"raw text to check. Found columns: {sorted(records[0])}. If this is a raw "
                "metadata file (e.g. from `everyvoice new`), rename its text column to "
                "'characters' (or 'phones'/'arpabet' if it's already phonemized)."
            )
        if "language" not in records[0]:
            if language is None:
                raise typer.BadParameter(
                    "--language is required for a psv file without a language column."
                )
            for record in records:
                record["language"] = language
    return records


def _default_styletts2_pretrained_symbols() -> Optional[list[str]]:
    """The default StyleTTS2 pretrained text-encoder symbol table, if the StyleTTS2
    submodule is available. Used as a fallback when a config declares a `pretrained`
    section without overriding `pretrained_symbols` explicitly."""
    try:
        from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.ev_config import (
            StyleTTS2PretrainedConfig,
        )
    except ImportError:
        return None
    return StyleTTS2PretrainedConfig().pretrained_symbols


def get_text_config_from_config_or_model(
    config: Optional[Path], model: Optional[Path]
) -> tuple["TextConfig", Optional[list[str]]]:
    """Helper for check_text_config: load a TextConfig from a config file or model file.

    Also returns the pretrained text-encoder symbol table declared alongside it,
    if any (currently only StyleTTS2 has one), used to suggest `to_replace`
    substitutions for symbols missing from it.
    """
    from everyvoice.config.text_config import TextConfig
    from everyvoice.config.utils import load_partials
    from everyvoice.utils import load_config_from_json_or_yaml_path, spinner

    pretrained_symbols: Optional[list[str]] = None
    if config:
        raw_config = load_config_from_json_or_yaml_path(config)
        if isinstance(raw_config, dict) and "VERSION" in raw_config:
            raw_config = load_partials(raw_config, ("text",), config_path=config)
            # 'text' has a default_factory, so a config that overrides neither
            # it nor path_to_text_config_file simply omits the key.
            text_config: TextConfig = TextConfig(**raw_config.get("text", {}))
            pretrained = raw_config.get("pretrained")
            if pretrained is not None:
                pretrained_symbols = (
                    pretrained.get("pretrained_symbols")
                    or _default_styletts2_pretrained_symbols()
                )
        else:
            text_config = TextConfig.load_config_from_path(config)
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
            ev_config = model_config["ev_config"]
            text_config = TextConfig(**ev_config["text"])
            pretrained_symbols = ev_config.get("pretrained", {}).get(
                "pretrained_symbols"
            )
        else:
            # Models without text config, e.g., a HiFiGan Vocoder, are not accepted here
            raise typer.BadParameter(
                f"Model/checkpoint '{model}' does not have an embedded text configuration."
            )
    return text_config, pretrained_symbols


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

    Inspect a text configuration for compatibility with an input file: test
    processing the file's text against the text configuration provided (or
    the text configuration found in a model), and report any characters or
    phones the file uses that aren't declared in the config.

    To instead check a config's declared symbols against a pretrained
    text-encoder's fixed symbol table (currently only StyleTTS2 has one), use
    `everyvoice check pretrained-symbols`.

    Required options: one of --config and --model, as well as one of --text-file and --psv-file.
    """
    from everyvoice.utils import spinner

    require_exactly_one_of(config, "--config", model, "--model")
    require_exactly_one_of(text_file, "--text-file", psv_file, "--psv-file")
    file_type = "text" if text_file else "psv"
    records = open_text_or_psv_file(text_file, psv_file, language)

    text_config, _ = get_text_config_from_config_or_model(config, model)

    # Expensive imports are deferred so we fail fast where we can
    with spinner("Loading software"):
        from everyvoice.preprocessor.preprocessor import Preprocessor
        from everyvoice.text.text_processor import TextProcessor
        from everyvoice.text.utils import guess_graphemes_in_text

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
            f"The following characters in your {file_type} file ('{text_file or psv_file}') are missing from your text config:",
            sorted(missing_characters),
        )
    if missing_phones:
        print(
            f"The following phones in your {file_type} file ('{text_file or psv_file}') are missing from your text config:",
            sorted(missing_phones),
        )
    if not missing_characters and not missing_phones:
        print(
            f"Every character and phone in your {file_type} file ('{text_file or psv_file}') "
            "is declared in your text config."
        )


@command(
    check_group,
    name="pretrained-symbols",
    short_help="Check a text config's declared symbols against a pretrained text-encoder's symbol table",
)
def check_pretrained_symbols(
    config: Annotated[
        Optional[Path],
        typer_file_option(
            "--config",
            "-c",
            help="path to a model config with a pretrained text-encoder, e.g. everyvoice-text-to-wav.yaml for StyleTTS2",
        ),
    ] = None,
    model: Annotated[
        Optional[Path],
        typer_file_option(
            "--model", "-m", help="path to a model whose config will be used"
        ),
    ] = None,
):
    """
    # Check Pretrained Symbols Help

    Some models (currently only StyleTTS2) use a pretrained text encoder with
    a fixed, frozen symbol table: every symbol your text config declares must
    be a member of that table, or the model can't produce a meaningful
    embedding for it. This command checks that, and for any symbol that isn't
    in the pretrained table, suggests the closest pretrained symbol to
    substitute it with — printed as a copy-pastable `to_replace` block.

    To instead check a text config against a sample of your text (are all the
    characters/phones your data uses declared in the config?), use
    `everyvoice check text-config`.

    Required options: one of --config and --model.
    """
    from everyvoice.utils import spinner

    require_exactly_one_of(config, "--config", model, "--model")

    text_config, pretrained_symbols = get_text_config_from_config_or_model(
        config, model
    )
    if pretrained_symbols is None:
        raise typer.BadParameter(
            "This config has no pretrained text-encoder symbol table to check "
            "against (currently only StyleTTS2 configs have one)."
        )

    # Expensive imports are deferred so we fail fast where we can
    with spinner("Loading software"):
        from everyvoice.text.text_processor import TextProcessor
        from everyvoice.text.utils import declared_content_symbols

    ev_symbols = declared_content_symbols(TextProcessor(text_config))
    pretrained_set = set(pretrained_symbols)
    missing_from_pretrained = sorted(s for s in ev_symbols if s not in pretrained_set)
    if not missing_from_pretrained:
        print(
            "All symbols declared in your text config are present in the "
            "pretrained text-encoder symbol table."
        )
        return

    with spinner("Computing symbol-mapping suggestions"):
        from everyvoice.text.utils_heavy import suggest_symbol_mapping

        result = suggest_symbol_mapping(ev_symbols, pretrained_symbols)
    print(
        "The following symbols declared in your text config are not present "
        "in the pretrained text-encoder symbol table:",
        missing_from_pretrained,
    )
    if result.suggestions:
        print(
            "Suggested substitutions — copy into your text config's "
            "'to_replace':\nto_replace:"
        )
        for symbol in missing_from_pretrained:
            target = result.suggestions.get(symbol)
            if target is not None:
                print(
                    f"  {symbol!r}: {target!r}  # distance={result.distances[symbol]:.2f}"
                )
    if result.unmapped:
        print(
            "No suitable pretrained replacement was found for:",
            sorted(result.unmapped),
        )
