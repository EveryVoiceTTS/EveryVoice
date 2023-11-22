from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Union

from loguru import logger
from pydantic import Field, FilePath, ValidationInfo, field_validator, model_validator

from everyvoice.config.shared_types import ConfigModel, PartialLoadConfig, init_context
from everyvoice.config.utils import (
    PossiblyRelativePath,
    PossiblySerializedCallable,
    load_partials,
)
from everyvoice.utils import generic_dict_loader, load_config_from_json_or_yaml_path


class AudioSpecTypeEnum(Enum):
    mel = "mel"  # TorchAudio implementation
    mel_librosa = "mel-librosa"  # Librosa implementation
    linear = "linear"  # TorchAudio Linear Spectrogram
    raw = "raw"  # TorchAudio Complex Spectrogram


class AudioConfig(ConfigModel):
    min_audio_length: float = 0.4
    max_audio_length: float = 11.0
    max_wav_value: float = 32767.0
    norm_db: float = -3.0
    sil_threshold: float = 1.0
    sil_duration: float = 0.1
    input_sampling_rate: int = 22050
    output_sampling_rate: int = 22050
    alignment_sampling_rate: int = 22050
    target_bit_depth: int = 16
    alignment_bit_depth: int = 16
    fft_window_frames: int = 1024
    fft_hop_frames: int = 256
    f_min: int = 0
    f_max: int = 8000
    n_fft: int = 1024
    n_mels: int = 80
    spec_type: Union[AudioSpecTypeEnum, str] = AudioSpecTypeEnum.mel_librosa.value
    vocoder_segment_size: int = 8192


class PitchCalculationMethod(Enum):
    pyworld = "pyworld"
    cwt = "cwt"


class Dataset(PartialLoadConfig):
    label: str = "YourDataSet"
    data_dir: PossiblyRelativePath = Path("/please/create/a/path/to/your/dataset/data")
    textgrid_dir: Union[PossiblyRelativePath, None] = None
    filelist: PossiblyRelativePath = Path(
        "/please/create/a/path/to/your/dataset/filelist"
    )
    filelist_loader: PossiblySerializedCallable = generic_dict_loader
    sox_effects: list = [["channels", "1"]]

    @field_validator(
        "data_dir",
        "textgrid_dir",
        "filelist",
    )
    @classmethod
    def relative_to_absolute(cls, value: Path, info: ValidationInfo) -> Path:
        return PartialLoadConfig.path_relative_to_absolute(value, info)


class PreprocessingConfig(PartialLoadConfig):
    dataset: str = "YourDataSet"
    pitch_type: Union[
        PitchCalculationMethod, str
    ] = PitchCalculationMethod.pyworld.value
    pitch_phone_averaging: bool = True
    energy_phone_averaging: bool = True
    value_separator: str = "--"
    train_split: float = Field(0.9, min=0.0, max=1.0)
    dataset_split_seed: int = 1234
    save_dir: PossiblyRelativePath = Path("./preprocessed/YourDataSet")
    audio: AudioConfig = Field(default_factory=AudioConfig)
    path_to_audio_config_file: Optional[FilePath] = None
    source_data: List[Dataset] = Field(default_factory=lambda: [Dataset()])

    @field_validator("save_dir", mode="before")
    @classmethod
    def relative_to_absolute(cls, value: Any, info: ValidationInfo) -> Path:
        if not isinstance(value, Path):
            try:
                value = Path(value)
            except TypeError as e:
                # Pydantic needs ValueErrors to raise its ValidationErrors
                raise ValueError from e

        absolute_dir = cls.path_relative_to_absolute(value, info)
        if not absolute_dir.exists():
            logger.info(f"Directory at {absolute_dir} does not exist. Creating...")
            absolute_dir.mkdir(parents=True, exist_ok=True)
        return absolute_dir

    @model_validator(mode="before")  # type: ignore
    def load_partials(self, info: ValidationInfo):
        config_path = (
            info.context.get("config_path", None) if info.context is not None else None
        )
        return load_partials(
            self,  # type: ignore
            ("audio",),
            config_path=config_path,
        )

    @staticmethod
    def load_config_from_path(path: Path) -> "PreprocessingConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        with init_context({"config_path": path}):
            config = PreprocessingConfig(**config)
        return config
