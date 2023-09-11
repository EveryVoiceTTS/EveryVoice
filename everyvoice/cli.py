from enum import Enum

import typer

from everyvoice.config import CONFIGS
from everyvoice.model.aligner.DeepForcedAligner.dfaligner.cli import (
    app as dfaligner_app,
)
from everyvoice.model.e2e.cli import app as e2e_app
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli import (
    app as fs2_app,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli import (
    preprocess as fs2_preprocess,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli import (
    synthesize as fs2_synthesize,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import app as hfgl_app

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
)

# Add subcommands from models
app.add_typer(
    dfaligner_app,
    name="align",
    help="Subcommands for the EveryVoice aligner",
    short_help="Subcommands for the EveryVoice aligner",
)
app.add_typer(
    e2e_app,
    name="text-to-wav",
    help="Subcommands for the EveryVoice end-to-end TTS model",
    short_help="Subcommands for the EveryVoice end-to-end TTS model",
)
app.add_typer(
    hfgl_app,
    name="spec-to-wav",
    help="Subcommands for the EveryVoice spec-to-wav model (aka Vocoder)",
    short_help="Subcommands for the EveryVoice spec-to-wav model (aka Vocoder)",
)
app.add_typer(
    fs2_app,
    name="text-to-spec",
    help="Subcommands for the EveryVoice text-to-spec model (aka Feature Prediction Network)",
    short_help="Subcommands for the EveryVoice text-to-spec model (aka Feature Prediction Network)",
)

# Add preprocess to root
app.command(
    short_help="Preprocess your data",
    help="This command will preprocess all of the data you need for use with EveryVoice. This is an alias for everyvoice text-to-spec preprocess.",
)(fs2_preprocess)

# Add synthesize to root
app.command(
    name="synthesize",
    short_help="Given some text and a text-to-spec model, generate some audio",
    help="Given some text and a text-to-spec model, generate some audio. This is an alias for everyvoice text-to-spec synthesize.",
)(fs2_synthesize)

_config_keys = {k: k for k in CONFIGS.keys()}

CONFIGS_ENUM = Enum("CONFIGS", _config_keys)  # type: ignore


@app.callback()
def callback():
    """
    # Welcome to the EveryVoice Command Line Interface

    Please visit the documentation to get started: [https://docs.everyvoice.ca](https://docs.everyvoice.ca)

    ## Use your own language/dataset

    To run the new dataset wizard please use the following command: everyvoice new-dataset

    ## Preprocess

    Once you have a configuration, preprocess your data by running everyvoice preprocess [OPTIONS]

    ## Train

    Once you have a configuration and have preprocessed your data, train a model by running everyvoice text-to-spec train [OPTIONS].
    EveryVoice has 4 different types of models you can train:

    1. **text-to-spec** --- this is the most common model you will need to train. It is a model from text inputs to spectral feature (aka spectrogram) outputs.

    2. **spec-to-wav** --- this is the model that turns your spectral features into audio. It is also known as a 'vocoder'. You will typically not need to train your own version. Please refer to [https://pathtocheckpoints](https://pathtocheckpoints) for more information.

    3. **text-to-wav** --- ....

    4. **align** --- ...

    ## Synthesize

    Once you have a trained model, generate some audio by running: everyvoice synthesize [OPTIONS]

    """


class TestSuites(str, Enum):
    all = "all"
    configs = "configs"
    dev = "dev"
    model = "model"
    preprocessing = "preprocessing"
    text = "text"


@app.command(
    help="""This command will help you create all the configuration necessary for using a new dataset.

In order to get started, please:
 - have your audio data (in .wav format) together in a folder
 - have a 'metadata' file that minimally has two columns, one for the text of the audio and one for the basename of the file

Inside /path/to/wavs/ you should have wav audio files like test0001.wav, test0002.wav etc - they can be called anything you want but the part of the file (minus the .wav portion) must be in your metadata file.

Example wavs directory: "/path/to/wavs/"

Example metadata: https://github.com/roedoejet/EveryVoice/blob/main/everyvoice/filelists/lj_full.psv

"""
)
def new_dataset():
    from everyvoice.wizard.main_tour import WIZARD_TOUR

    WIZARD_TOUR.run()


@app.command(hidden=True)
def test(suite: TestSuites = typer.Argument(TestSuites.dev)):
    """Run a test suite"""
    from everyvoice.run_tests import run_tests

    run_tests(suite)


CLICK_APP = typer.main.get_group(app)

if __name__ == "__main__":
    app()
