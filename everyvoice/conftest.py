"""
Fixtures shared by all unit tests.
Placed in everyvoice instead of everyvoice/tests so that submodules also see it.

This gets read by pytest every time tests start, so other test configuration
settings/overrides can be inserted here.
"""

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pytest import fixture

if TYPE_CHECKING:
    from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
        FastSpeech2,
    )
    from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import HiFiGAN


# On Windows, redirecting pytest output to file happens in the system encoding, which
# is not utf-8 by default. We don't like that, so fix it.
if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore

# Stabilize typer help output to a consistent width for unit testing purposes
if os.name == "nt":
    os.environ["COLUMNS"] = "100"
else:
    os.environ["COLUMNS"] = "99"
os.environ["NO_COLOR"] = "1"  # disable Typer help colouring for unit tests


@fixture(scope="session")
def dummy_models(tmp_path_factory) -> tuple["FastSpeech2", Path, "HiFiGAN", Path]:
    from .tests.model_stubs import get_dummy_models

    return get_dummy_models(tmp_path_factory.mktemp("dummy_models"))


@fixture(scope="session")
def dummy_fp_path(dummy_models) -> Path:
    return dummy_models[1]


@fixture(scope="session")
def dummy_vocoder_path(dummy_models) -> Path:
    return dummy_models[3]


@fixture(scope="session")
def stubbed_model(dummy_models) -> tuple["FastSpeech2", Path]:
    dummy_fp, dummy_fp_path, dummy_vocoder, dummy_vocoder_path = dummy_models
    return dummy_fp, dummy_fp_path


@fixture(scope="session")
def stubbed_vocoder(dummy_models) -> tuple["HiFiGAN", Path]:
    dummy_fp, dummy_fp_path, dummy_vocoder, dummy_vocoder_path = dummy_models
    return dummy_vocoder, dummy_vocoder_path
