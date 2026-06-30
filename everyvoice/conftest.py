"""
Fixtures shared by all unit tests.
Places in everyvoice instead of everyvoice/tests so that submodules also see it.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from pytest import fixture

if TYPE_CHECKING:
    from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
        FastSpeech2,
    )
    from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import HiFiGAN


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
