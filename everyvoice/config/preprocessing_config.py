import contextlib
from enum import Enum
from pathlib import Path
from typing import Callable, List, Union

from loguru import logger
from pydantic import DirectoryPath, Field, FilePath, validator

from everyvoice.config.shared_types import ConfigModel, PartialConfigModel
from everyvoice.config.utils import convert_callables, convert_paths
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
    spec_type: AudioSpecTypeEnum = AudioSpecTypeEnum.mel_librosa
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
    sox_effects: list = []

    @convert_callables(kwargs_to_convert=["filelist_loader"])
    @convert_paths(kwargs_to_convert=["data_dir", "textgrid_dir", "filelist"])
    def __init__(
        self,
        **data,
    ) -> None:
        """Custom init to process file paths"""
        super().__init__(
            **data,
            expandable=["sox_effects"],
        )


class PreprocessingConfig(PartialConfigModel):
    dataset: str = "YourDataSet"
    pitch_type: PitchCalculationMethod = PitchCalculationMethod.pyworld
    pitch_phone_averaging: bool = True
    energy_phone_averaging: bool = True
    value_separator: str = "--"
    save_dir: DirectoryPath = Path("./preprocessed/YourDataSet")
    audio: AudioConfig = Field(default_factory=AudioConfig)
    source_data: List[Dataset] = Field(default_factory=lambda: [Dataset()])

    @validator("save_dir")
    def create_dir(cls, v: str):
        return v

    @convert_paths(kwargs_to_convert=["save_dir"])
    def __init__(self, **data) -> None:
        """Custom init to process file paths"""
        # Supress keyerrors because defaults will be used if not supplied
        with contextlib.suppress(KeyError):
            if not data["save_dir"].exists():
                logger.info(
                    f"Directory at {data['save_dir']} does not exist. Creating..."
                )
                data["save_dir"].mkdir(parents=True, exist_ok=True)
        super().__init__(**data, expandable=["audio", "source_data"])

    @staticmethod
    def load_config_from_path(path: Path) -> "PreprocessingConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        return PreprocessingConfig(**config)
