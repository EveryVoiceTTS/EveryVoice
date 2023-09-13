from enum import Enum

import typer

from everyvoice.config import CONFIGS
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli import (
    preprocess as fs2_preprocess,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli import (
    synthesize as fs2_synthesize,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli import (
    train as fs2_train,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import (
    synthesize as hfg_synthesize,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import train as hfg_train

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
)


class ModelTypes(str, Enum):
    text_to_spec = "text-to-spec"
    spec_to_wav = "spec-to-wav"


# Add preprocess to root
app.command(
    short_help="Preprocess your data",
    help="This command will preprocess all of the data you need for use with EveryVoice.",
)(fs2_preprocess)

# Add the train commands
train_group = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
)

train_group.command(
    name="text-to-spec",
    short_help="Train your Text-to-Spec model",
)(fs2_train)

train_group.command(
    name="spec-to-wav",
    short_help="Train your Spec-to-Wav model",
)(hfg_train)

app.add_typer(
    train_group,
    name="train",
    help="Train your EveryVoice models",
    short_help="Train your EveryVoice models",
)

# Add synthesize commands
synthesize_group = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
)

synthesize_group.command(
    name="text-to-spec",
    short_help="Given some text and a text-to-spec model, generate some audio",
    help="Given some text and a text-to-spec model, generate some audio.",
)(fs2_synthesize)

synthesize_group.command(
    name="spec-to-wav",
    short_help="Given some Mel spectrograms and a spec-to-wav model, generate some audio",
    help="Given some Mel spectrograms and a spec-to-wav model, generate some audio.",
)(hfg_synthesize)

app.add_typer(
    train_group,
    name="synthesize",
    help="Synthesize using your pre-trained EveryVoice models",
    short_help="Synthesize using your pre-trained EveryVoice models",
)


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

    Once you have a configuration and have preprocessed your data, train a model by running everyvoice train [text-to-spec|spec-to-wav] [OPTIONS].
    EveryVoice has different types of models you can train:

    1. **text-to-spec** --- this is the most common model you will need to train. It is a model from text inputs to spectral feature (aka spectrogram) outputs.

    2. **spec-to-wav** --- this is the model that turns your spectral features into audio. It is also known as a 'vocoder'. You will typically not need to train your own version. Please refer to [https://pathtocheckpoints](https://pathtocheckpoints) for more information.

    ## Synthesize

    Once you have a trained model, generate some audio by running: everyvoice synthesize [text-to-spec|spec-to-wav] [OPTIONS]

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
