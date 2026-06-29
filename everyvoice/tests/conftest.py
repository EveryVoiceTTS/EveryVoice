from pathlib import Path

from pytest import fixture

from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
    FastSpeech2,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import HiFiGAN
from everyvoice.tests.model_stubs import get_stubbed_model, get_stubbed_vocoder


@fixture(scope="session")
def stubbed_model(tmp_path_factory) -> tuple[FastSpeech2, Path]:
    return get_stubbed_model(tmp_path_factory.mktemp("vocoder"))


@fixture(scope="session")
def stubbed_vocoder(tmp_path_factory) -> tuple[HiFiGAN, Path]:
    return get_stubbed_vocoder(tmp_path_factory.mktemp("vocoder"))
