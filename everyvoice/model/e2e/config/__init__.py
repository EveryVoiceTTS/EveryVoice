from pathlib import Path
from typing import Optional, Union

from pydantic import Field, FilePath, ValidationInfo, model_validator

from everyvoice.config.shared_types import (
    BaseModelWithContact,
    BaseTrainingConfig,
    ContactInformation,
    init_context,
)
from everyvoice.config.utils import PossiblyRelativePath, load_partials
from everyvoice.model.aligner.config import AlignerConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.utils import load_config_from_json_or_yaml_path


# The contact information only needs to be registered on the main config
class AlignerConfigNoContact(AlignerConfig):
    contact: Optional[ContactInformation] = None  # type: ignore


class VocoderConfigNoContact(VocoderConfig):
    contact: Optional[ContactInformation] = None  # type: ignore


class FeaturePredictionConfigNoContact(FeaturePredictionConfig):
    contact: Optional[ContactInformation] = None  # type: ignore


class E2ETrainingConfig(BaseTrainingConfig):
    feature_prediction_checkpoint: Union[None, PossiblyRelativePath] = None
    vocoder_checkpoint: Union[None, PossiblyRelativePath] = None


class EveryVoiceConfig(BaseModelWithContact):
    aligner: AlignerConfig | AlignerConfigNoContact = Field(
        default_factory=AlignerConfigNoContact  # type: ignore
    )
    path_to_aligner_config_file: Optional[FilePath] = None

    feature_prediction: FeaturePredictionConfig | FeaturePredictionConfigNoContact = (
        Field(default_factory=FeaturePredictionConfigNoContact)  # type: ignore
    )
    path_to_feature_prediction_config_file: Optional[FilePath] = None

    vocoder: VocoderConfig | VocoderConfigNoContact = Field(
        default_factory=VocoderConfigNoContact  # type: ignore
    )
    path_to_vocoder_config_file: Optional[FilePath] = None

    training: E2ETrainingConfig = Field(default_factory=E2ETrainingConfig)
    path_to_training_config_file: Optional[FilePath] = None

    @model_validator(mode="before")  # type: ignore
    def load_partials(self, info: ValidationInfo):
        config_path = (
            info.context.get("config_path", None) if info.context is not None else None
        )
        return load_partials(
            self,  # type: ignore
            ("aligner", "feature_prediction", "vocoder", "training"),
            config_path=config_path,
        )

    @staticmethod
    def load_config_from_path(
        path: Path,
    ) -> "EveryVoiceConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        with init_context({"config_path": path}):
            config = EveryVoiceConfig(**config)
        return config
