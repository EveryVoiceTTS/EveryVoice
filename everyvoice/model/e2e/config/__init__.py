from pathlib import Path
from typing import Union

from pydantic import Field, FilePath, validator
from pydantic.fields import ModelField

from everyvoice.config.shared_types import BaseTrainingConfig, PartialConfigModel
from everyvoice.model.aligner.config import AlignerConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.utils import load_config_from_json_or_yaml_path, rel_path_to_abs_path


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
    aligner: AlignerConfig = Field(default_factory=AlignerConfig)
    feature_prediction: FeaturePredictionConfig = Field(
        default_factory=FeaturePredictionConfig
    )
    vocoder: VocoderConfig = Field(default_factory=VocoderConfig)
    training: E2ETrainingConfig = Field(default_factory=E2ETrainingConfig)

    @staticmethod
    def load_config_from_path(
        path: Path,
    ) -> "EveryVoiceConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        return EveryVoiceConfig(**config)
