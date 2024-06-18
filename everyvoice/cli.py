import json
from enum import Enum
from pathlib import Path
from typing import Any

import typer

from everyvoice._version import VERSION
from everyvoice.base_cli.checkpoint import inspect as inspect_checkpoint
from everyvoice.base_cli.interfaces import complete_path
from everyvoice.model.aligner.wav2vec2aligner.aligner.cli import (
    align_single as ctc_segment,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.preprocess import (
    preprocess as preprocess_fs2,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.synthesize import (
    synthesize as synthesize_fs2,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.train import (
    train as train_fs2,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import (
    synthesize as synthesize_hfg,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import train as train_hfg
from everyvoice.wizard import (
    ALIGNER_CONFIG_FILENAME_PREFIX,
    PREPROCESSING_CONFIG_FILENAME_PREFIX,
    SPEC_TO_WAV_CONFIG_FILENAME_PREFIX,
    TEXT_CONFIG_FILENAME_PREFIX,
    TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX,
    TEXT_TO_WAV_CONFIG_FILENAME_PREFIX,
)


# See https://github.com/tiangolo/typer/issues/428#issuecomment-1238866548
class TyperGroupOrderAsDeclared(typer.core.TyperGroup):
    def list_commands(self, ctx):
        return self.commands.keys()


app = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
    cls=TyperGroupOrderAsDeclared,
    help="""
    # Welcome to the EveryVoice Command Line Interface

    Please visit the documentation to get started: [https://docs.everyvoice.ca](https://docs.everyvoice.ca)

    ## Use your own language/dataset

    To run the wizard for a new project please use the following command: everyvoice new-project

    ## Segment long files in your dataset

    If you have long audio files that contain more than one utterance,
    you can use the segmentation tool by running everyvoice segment [OPTIONS]

    ## Preprocess

    Once you have a configuration, preprocess your data by running everyvoice preprocess [OPTIONS]

    ## Train

    Once you have a configuration and have preprocessed your data, train a model by running everyvoice train [text-to-spec|spec-to-wav] [OPTIONS].
    EveryVoice has different types of models you can train:

    1. **text-to-spec** --- this is the most common model you will need to train. It is a model from text inputs to spectral feature (aka spectrogram) outputs.

    2. **spec-to-wav** --- this is the model that turns your spectral features into audio. It is also known as a 'vocoder'. You will typically not need to train your own version. Please refer to [https://pathtocheckpoints](https://pathtocheckpoints) for more information.

    ## Synthesize

    Once you have a trained model, generate some audio by running: everyvoice synthesize [text-to-spec|spec-to-wav] [OPTIONS]
""",
)


class ModelTypes(str, Enum):
    text_to_spec = "text-to-spec"
    spec_to_wav = "spec-to-wav"


app.command(
    short_help="Segment a long audio file",
    name="segment",
    help="""
    # Segmentation help

    This command will segment a long audio file into multiple utterances which is required for training a TTS system.
    This command should work on most languages and you should run it before running the new project or preprocessing steps.
    """,
)(ctc_segment)


@app.command(
    short_help="This command will help you create all the configuration necessary for using a new project.",
    help="""
    # This command will help you create all the configuration necessary for using a new project.

    ## Getting Started

    In order to get started, please:

        1. make sure your audio data is available on your computer and in .wav format

        2. have a 'metadata' file that minimally has two columns, one for the text of the audio and one for the basename of the file


    ## Extra details and examples

    Inside /path/to/wavs/ you should have wav audio files like test0001.wav, test0002.wav etc - they can be called anything you want but the part of the file (minus the .wav portion) must be in your metadata file.

    Example wavs directory: "/path/to/wavs/"

    Example metadata: [https://github.com/roedoejet/EveryVoice/blob/main/everyvoice/filelists/lj_full.psv](https://github.com/roedoejet/EveryVoice/blob/main/everyvoice/filelists/lj_full.psv)

""",
)
def new_project():
    from everyvoice.wizard.main_tour import get_main_wizard_tour

    get_main_wizard_tour().run()


# Add preprocess to root
app.command(
    short_help="Preprocess your data",
    help=f"""
    # Preprocess Help

    This command will preprocess all of the data you need for use with EveryVoice.

    By default every step of the preprocessor will be done by running:
    \n\n
    **everyvoice preprocess config/{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml**
    \n\n
    If you only want to process specific things, you can run specific commands by adding them as options for example:
    \n\n
    **everyvoice preprocess config/{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml -s energy -s pitch**
    """,
)(preprocess_fs2)

# Add the train commands
train_group = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
    cls=TyperGroupOrderAsDeclared,
    help="""
    # Train Help

        - **text-to-spec** --- this is the most common model you will need to train. It is a model from text inputs to spectral feature (aka spectrogram) outputs.

        - **spec-to-wav** --- this is the model that turns your spectral features into audio. It is also known as a 'vocoder'. You will typically not need to train your own version. Please refer to [https://pathtocheckpoints](https://pathtocheckpoints) for more information.
    """,
)

train_group.command(
    name="text-to-spec",
    short_help="Train your Text-to-Spec model",
    help=f"""Train your text-to-spec model.  For example:

    **everyvoice train text-to-spec config/{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml**
    """,
)(train_fs2)

train_group.command(
    name="spec-to-wav",
    short_help="Train your Spec-to-Wav model",
    help=f"""Train your spec-to-wav model. For example:

    **everyvoice train spec-to-wav config/{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml**
    """,
)(train_hfg)

app.add_typer(
    train_group,
    name="train",
    short_help="Train your EveryVoice models",
)

# Add synthesize commands
synthesize_group = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
    cls=TyperGroupOrderAsDeclared,
    help="""
    # Synthesize Help

        - **from-text** --- This is the most common input for performing normal speech synthesis. It will take text or a filelist with text and produce either waveform audio or spectrogram.

        - **from-spec** --- This is the model that turns your spectral features into audio. This type of synthesis is also known as copy synthesis and unless you know what you are doing, you probably don't want to do this.
    """,
)

synthesize_group.command(
    name="from-text",
)(synthesize_fs2)

synthesize_group.command(
    name="from-spec",
)(synthesize_hfg)

app.add_typer(
    synthesize_group,
    name="synthesize",
    short_help="Synthesize using your pre-trained EveryVoice models",
)

app.command(
    name="inspect-checkpoint",
    short_help="Extract structural information from a checkpoint",
)(inspect_checkpoint)


class TestSuites(str, Enum):
    all = "all"
    cli = "cli"
    config = "config"
    dev = "dev"
    model = "model"
    preprocessing = "preprocessing"
    text = "text"
    fs2 = "fs2"


@app.command(hidden=True)
def test(suite: TestSuites = typer.Argument(TestSuites.dev)):
    """Run a test suite"""
    from everyvoice.run_tests import run_tests

    run_tests(suite)


# Deferred full initialization to optimize the CLI, but still exposed for unit testing.
SCHEMAS_TO_OUTPUT: dict[str, Any] = {}  # dict[str, type[BaseModel]]


@app.command()
def demo(
    text_to_spec_model: Path = typer.Argument(
        ...,
        file_okay=True,
        exists=True,
        dir_okay=False,
        help="The path to a trained text-to-spec EveryVoice model.",
        autocompletion=complete_path,
    ),
    spec_to_wav_model: Path = typer.Argument(
        ...,
        help="The path to a trained vocoder.",
        dir_okay=False,
        file_okay=True,
        autocompletion=complete_path,
    ),
    output_dir: Path = typer.Option(
        "synthesis_output",
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="The directory where your synthesized audio should be written",
        autocompletion=complete_path,
    ),
    accelerator: str = typer.Option("auto", "--accelerator", "-a"),
):
    from everyvoice.demo.app import create_demo_app

    demo = create_demo_app(
        text_to_spec_model_path=text_to_spec_model,
        spec_to_wav_model_path=spec_to_wav_model,
        output_dir=output_dir,
        accelerator=accelerator,
    )
    demo.launch()


@app.command(hidden=True)
def update_schemas(
    out_dir: Path = typer.Option(
        None,
        "-o",
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        exists=True,
        autocompletion=complete_path,
    ),
):
    """Update the JSON Schemas. This is hidden because you shouldn't be calling this unless you are
    a developer for EveryVoice. Note: Pydantic will raise some Warnings related to the Callable fields
    having string Schemas. These can be ignored.
    """
    if out_dir is None:
        schema_dir_path = Path(__file__).parent / ".schema"
    else:
        schema_dir_path = out_dir

    # Defer somewhat slow imports to optimize CLI
    from everyvoice.config.preprocessing_config import PreprocessingConfig
    from everyvoice.config.text_config import TextConfig
    from everyvoice.model.aligner.config import AlignerConfig
    from everyvoice.model.e2e.config import EveryVoiceConfig
    from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
    from everyvoice.model.vocoder.config import VocoderConfig

    # We should not be changing the schema for patches, so only include major/minor version
    MAJOR_MINOR_VERSION = ".".join(VERSION.split(".")[:2])

    SCHEMAS_TO_OUTPUT.update(
        {
            f"{ALIGNER_CONFIG_FILENAME_PREFIX}-schema-{MAJOR_MINOR_VERSION}.json": AlignerConfig,
            f"{TEXT_TO_WAV_CONFIG_FILENAME_PREFIX}-schema-{MAJOR_MINOR_VERSION}.json": EveryVoiceConfig,
            f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}-schema-{MAJOR_MINOR_VERSION}.json": FeaturePredictionConfig,
            f"{PREPROCESSING_CONFIG_FILENAME_PREFIX}-schema-{MAJOR_MINOR_VERSION}.json": PreprocessingConfig,
            f"{TEXT_CONFIG_FILENAME_PREFIX}-schema-{MAJOR_MINOR_VERSION}.json": TextConfig,
            f"{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}-schema-{MAJOR_MINOR_VERSION}.json": VocoderConfig,
        }
    )

    for filename, schema in SCHEMAS_TO_OUTPUT.items():
        if (schema_dir_path / filename).exists():
            raise FileExistsError(
                f"Sorry a schema already exists for version {filename}.\n"
                "If it's already been published to the schema store, please bump the EveryVoice minor version number and generate the schemas again.\n"
                "If the current minor version is still in development, just delete the schema files and try again."
            )
        with open(schema_dir_path / filename, "w") as f:
            json.dump(schema.model_json_schema(), f, indent=2)
            f.write("\n")


CLICK_APP = typer.main.get_group(app)

if __name__ == "__main__":
    app()
