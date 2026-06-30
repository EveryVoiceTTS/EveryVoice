from pathlib import Path
from typing import TYPE_CHECKING

from pytest import fixture

from everyvoice.tests.model_stubs import (
    get_dummy_models,
    get_stubbed_model,
    get_stubbed_vocoder,
)

if TYPE_CHECKING:
    from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
        FastSpeech2,
    )
    from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import HiFiGAN


@fixture(scope="session")
def stubbed_model(tmp_path_factory) -> tuple["FastSpeech2", Path]:
    return get_stubbed_model(tmp_path_factory.mktemp("vocoder"))


@fixture(scope="session")
def stubbed_vocoder(tmp_path_factory) -> tuple["HiFiGAN", Path]:
    return get_stubbed_vocoder(tmp_path_factory.mktemp("vocoder"))


@fixture(scope="session")
def dummy_models(tmp_path_factory) -> tuple[Path, Path]:
    return get_dummy_models(tmp_path_factory.mktemp("dummy_models"))


@fixture(scope="session")
def dummy_fp_path(dummy_models) -> Path:
    return dummy_models[0]


@fixture(scope="session")
def dummy_vocoder_path(dummy_models) -> Path:
    return dummy_models[1]
