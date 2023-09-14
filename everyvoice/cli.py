from enum import Enum

import typer

from everyvoice.config import CONFIGS
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli import (
    preprocess as preprocess_fs2,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli import (
    synthesize as synthesize_fs2,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli import (
    train as train_fs2,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import (
    synthesize as synthesize_hfg,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import train as train_hfg


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

    To run the new dataset wizard please use the following command: everyvoice new-dataset

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


@app.command(
    short_help="This command will help you create all the configuration necessary for using a new dataset.",
    help="""
    # This command will help you create all the configuration necessary for using a new dataset.

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
def new_dataset():
    from everyvoice.wizard.main_tour import WIZARD_TOUR

    WIZARD_TOUR.run()


# Add preprocess to root
app.command(
    short_help="Preprocess your data",
    help="""
    # Preprocess Help

    This command will preprocess all of the data you need for use with EveryVoice.

    By default every step of the preprocessor will be done, but you can run specific commands by adding them as arguments or --steps options for example: **everyvoice preprocess energy pitch** or **everyvoice preprocess --steps energy --steps audio**
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
)(train_fs2)

train_group.command(
    name="spec-to-wav",
    short_help="Train your Spec-to-Wav model",
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

        - **text-to-spec** --- this is the most common model to run

        - **spec-to-wav** --- this is the model that turns your spectral features into audio. It is also known as a 'vocoder'. You will typically not need to train your own version. Please refer to [https://pathtocheckpoints](https://pathtocheckpoints) for more information.
    """,
)

synthesize_group.command(
    name="text-to-wav",
    short_help="Given some text and a trained model, generate some audio",
    help="Given some text and a trained model, generate some audio.",
)(synthesize_fs2)

synthesize_group.command(
    name="spec-to-wav",
    short_help="Given some Mel spectrograms and a trained model, generate some audio",
    help="Given some Mel spectrograms and a trained model, generate some audio.",
)(synthesize_hfg)

app.add_typer(
    synthesize_group,
    name="synthesize",
    short_help="Synthesize using your pre-trained EveryVoice models",
)


_config_keys = {k: k for k in CONFIGS.keys()}

CONFIGS_ENUM = Enum("CONFIGS", _config_keys)  # type: ignore


class TestSuites(str, Enum):
    all = "all"
    configs = "configs"
    dev = "dev"
    model = "model"
    preprocessing = "preprocessing"
    text = "text"


@app.command(hidden=True)
def test(suite: TestSuites = typer.Argument(TestSuites.dev)):
    """Run a test suite"""
    from everyvoice.run_tests import run_tests

    run_tests(suite)


CLICK_APP = typer.main.get_group(app)

if __name__ == "__main__":
    app()
