from pathlib import Path
from typing import Dict, Union

from pydantic import FilePath

from smts.config.shared_types import BaseTrainingConfig, PartialConfigModel
from smts.config.utils import __file__ as config_dir
from smts.config.utils import convert_paths
from smts.model.aligner.config import AlignerConfig
from smts.model.feature_prediction.config import FeaturePredictionConfig
from smts.model.vocoder.config import VocoderConfig
from smts.utils import load_config_from_json_or_yaml_path, return_configs_from_dir


class E2ETrainingConfig(BaseTrainingConfig):
    feature_prediction_checkpoint: Union[None, FilePath]
    vocoder_checkpoint: Union[None, FilePath]

    @convert_paths(
        kwargs_to_convert=["feature_prediction_checkpoint", "vocoder_checkpoint"]
    )
    def __init__(self, **data) -> None:
        """Custom init to process file paths"""
        super().__init__(
            **data,
        )


class SMTSConfig(PartialConfigModel):
    aligner: AlignerConfig
    feature_prediction: FeaturePredictionConfig
    vocoder: VocoderConfig
    training: E2ETrainingConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def load_config_from_path(
        path: Path = (Path(config_dir).parent / "base" / "base_composed.yaml"),
    ) -> "SMTSConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        return SMTSConfig(**config)


CONFIG_DIR = Path(__file__).parent
CONFIGS: Dict[str, Path] = return_configs_from_dir(CONFIG_DIR)
