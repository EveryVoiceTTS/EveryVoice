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
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import app as hfgl_app

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(dfaligner_app, name="dfa")
app.add_typer(e2e_app, name="e2e")
app.add_typer(hfgl_app, name="hifigan")
app.add_typer(fs2_app, name="fs2")

_config_keys = {k: k for k in CONFIGS.keys()}

CONFIGS_ENUM = Enum("CONFIGS", _config_keys)  # type: ignore


class TestSuites(str, Enum):
    all = "all"
    configs = "configs"
    dev = "dev"
    model = "model"
    preprocessing = "preprocessing"
    text = "text"


@app.command(
    help="""This command will help you create all the configuration necessary for your using your dataset.

In order to get started, please:
 - have your audio data (in .wav format) together in a folder
 - have a 'metadata' file that minimally has two columns, one for the text of the audio and one for the basename of the file

Inside /path/to/wavs/ you should have wav audio files like test0001.wav, test0002.wav etc - they can be called anything you want but the part of the file (minus the .wav portion) must be in your metadata file.

Example wavs directory: "/path/to/wavs/"

Example metadata: https://github.com/roedoejet/EveryVoice/blob/main/everyvoice/filelists/lj_full.psv

"""
)
def config_wizard():
    from everyvoice.wizard.main_tour import WIZARD_TOUR

    WIZARD_TOUR.run()


@app.command()
def test(suite: TestSuites = typer.Argument(TestSuites.dev)):
    """This command will run the test suite specified by the user"""
    from everyvoice.run_tests import run_tests

    run_tests(suite)


CLICK_APP = typer.main.get_group(app)

if __name__ == "__main__":
    app()
