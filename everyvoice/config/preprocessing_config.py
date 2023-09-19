import contextlib
from enum import Enum
from pathlib import Path
from typing import Callable, List, Union

from loguru import logger
from pydantic import DirectoryPath, Field, FilePath, validator
from pydantic.fields import ModelField

from everyvoice.config.shared_types import ConfigModel, PartialConfigModel
from everyvoice.config.utils import string_to_callable
from everyvoice.utils import (
    generic_dict_loader,
    load_config_from_json_or_yaml_path,
    rel_path_to_abs_path,
)


class AudioSpecTypeEnum(Enum):
    mel = "mel"  # TorchAudio implementation
    mel_librosa = "mel-librosa"  # Librosa implementation
    linear = "linear"  # TorchAudio Linear Spectrogram
    raw = "raw"  # TorchAudio Complex Spectrogram


class AudioConfig(ConfigModel):
    min_audio_length: float = 0.25
    max_audio_length: float = 11.0
    max_wav_value: float = 32768.0
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
    kaldi = "kaldi"
    cwt = "cwt"


class Dataset(PartialConfigModel):
    label: str = "YourDataSet"
    data_dir: Union[DirectoryPath, Path] = Path(
        "/please/create/a/path/to/your/dataset/data"
    )
    textgrid_dir: Union[DirectoryPath, None] = None
    filelist: Union[FilePath, Path] = Path(
        "/please/create/a/path/to/your/dataset/filelist"
    )
    filelist_loader: Callable = generic_dict_loader
    sox_effects: list = [["channels", "1"]]

    @validator("filelist_loader", pre=True, always=True)
    def convert_callable_filelist_loader(cls, v, values):
        func = string_to_callable(v)
        values["filelist_loader"] = func
        return func


class PreprocessingConfig(PartialConfigModel):
    dataset: str = "YourDataSet"
    pitch_type: Union[
        PitchCalculationMethod, str
    ] = PitchCalculationMethod.pyworld.value
    pitch_phone_averaging: bool = True
    energy_phone_averaging: bool = True
    value_separator: str = "--"
    train_split: float = 0.9
    dataset_split_seed: int = 1234
    # save_dir used to be a DirectoryPath but pydantic as some magic that
    # demand that the directory exists but it doesn't take into account that
    # fact that we need it to be relative to the config file.
    save_dir: Path = Path("./preprocessed/YourDataSet")
    audio: AudioConfig = Field(default_factory=AudioConfig)
    source_data: List[Dataset] = Field(default_factory=lambda: [Dataset()])

    @staticmethod
    def load_config_from_path(path: Path) -> "PreprocessingConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)

        config["save_dir"] = (path.parent / config["save_dir"]).resolve()
        for data in source_data:
            data["data_dir"] = (path.parent / config["data_dir"]).resolve()
            data["filelist"] = (path.parent / config["filelist"]).resolve()
            if data["textgrid_dir"] is not None:
                data["textgrid_dir"] = (path.parent / data["textgrid_dir"]).resolve()
        return PreprocessingConfig(**config)
