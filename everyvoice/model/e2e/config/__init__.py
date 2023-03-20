from pathlib import Path
from typing import Dict, Union

from pydantic import FilePath, validator
from pydantic.fields import ModelField

from everyvoice.config.shared_types import BaseTrainingConfig, PartialConfigModel
from everyvoice.config.utils import __file__ as config_dir
from everyvoice.model.aligner.config import AlignerConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.utils import (
    load_config_from_json_or_yaml_path,
    rel_path_to_abs_path,
    return_configs_from_dir,
)


class E2ETrainingConfig(BaseTrainingConfig):
    feature_prediction_checkpoint: Union[None, FilePath]
    vocoder_checkpoint: Union[None, FilePath]

    @validator(
        "feature_prediction_checkpoint", "vocoder_checkpoint", pre=True, always=True
    )
    def convert_paths(cls, v, values, field: ModelField):
        path = rel_path_to_abs_path(v)
        values[field.name] = path
        return path


class EveryVoiceConfig(PartialConfigModel):
    aligner: AlignerConfig
    feature_prediction: FeaturePredictionConfig
    vocoder: VocoderConfig
    training: E2ETrainingConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def load_config_from_path(
        path: Path = (Path(config_dir).parent / "base" / "base_composed.yaml"),
    ) -> "EveryVoiceConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        return EveryVoiceConfig(**config)


CONFIG_DIR = Path(__file__).parent
CONFIGS: Dict[str, Path] = return_configs_from_dir(CONFIG_DIR)
