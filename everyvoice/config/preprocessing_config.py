import contextlib
from enum import Enum
from pathlib import Path
from typing import List, Union

from loguru import logger
from pydantic import Field, field_validator

from everyvoice.config.shared_types import ConfigModel, PartialConfigModel
from everyvoice.config.utils import PossiblyRelativePath, PossiblySerializedCallable
from everyvoice.utils import generic_dict_loader, load_config_from_json_or_yaml_path


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
    data_dir: PossiblyRelativePath = Path("/please/create/a/path/to/your/dataset/data")
    textgrid_dir: Union[PossiblyRelativePath, None] = None
    filelist: PossiblyRelativePath = Path(
        "/please/create/a/path/to/your/dataset/filelist"
    )
    filelist_loader: PossiblySerializedCallable = generic_dict_loader
    sox_effects: list = [["channels", "1"]]


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
    save_dir: PossiblyRelativePath = Path("./preprocessed/YourDataSet")
    audio: AudioConfig = Field(default_factory=AudioConfig)
    source_data: List[Dataset] = Field(default_factory=lambda: [Dataset()])

    @field_validator("save_dir", mode="after")
    def create_dir(cls, value: Path):
        # Supress keyerrors because defaults will be used if not supplied
        with contextlib.suppress(KeyError):
            if not value.exists():
                logger.info(f"Directory at {value} does not exist. Creating...")
                value.mkdir(parents=True, exist_ok=True)
        return value

    @staticmethod
    def load_config_from_path(path: Path) -> "PreprocessingConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        return PreprocessingConfig(**config)
