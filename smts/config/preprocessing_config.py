from enum import Enum
from pathlib import Path
from typing import Callable, List, Union

from loguru import logger
from pydantic import DirectoryPath, FilePath, validator

from smts.config.shared_types import ConfigModel, PartialConfigModel
from smts.config.utils import convert_callables, convert_paths


class AudioSpecTypeEnum(Enum):
    mel = "mel"  # TorchAudio implementation
    mel_librosa = "mel-librosa"  # Librosa implementation
    linear = "linear"  # TorchAudio Linear Spectrogram
    raw = "raw"  # TorchAudio Complex Spectrogram


class AudioConfig(ConfigModel):
    min_audio_length: float
    max_audio_length: float
    max_wav_value: float
    norm_db: float
    sil_threshold: float
    sil_duration: float
    input_sampling_rate: int
    output_sampling_rate: int
    alignment_sampling_rate: int
    target_bit_depth: int
    alignment_bit_depth: int
    fft_window_frames: int
    fft_hop_frames: int
    f_min: int
    f_max: int
    n_fft: int
    n_mels: int
    spec_type: AudioSpecTypeEnum
    vocoder_segment_size: int


class PitchCalculationMethod(Enum):
    pyworld = "pyworld"
    kaldi = "kaldi"
    cwt = "cwt"


class Dataset(PartialConfigModel):
    label: str
    data_dir: DirectoryPath
    textgrid_dir: Union[DirectoryPath, None]
    filelist: Union[FilePath, Path]
    filelist_loader: Callable
    sox_effects: list

    @convert_callables(kwargs_to_convert=["filelist_loader"])
    @convert_paths(kwargs_to_convert=["data_dir", "textgrid_dir", "filelist"])
    def __init__(
        self,
        **data,
    ) -> None:
        """Custom init to process file paths"""
        if not data["filelist"].exists():
            logger.warning(
                f"Filelist {data['filelist']} does not exist. If you're just preprocessing, that's fine, otherwise this will cause an error"
            )
        super().__init__(
            **data,
            expandable=["sox_effects"],
        )


class PreprocessingConfig(PartialConfigModel):
    dataset: str
    pitch_type: PitchCalculationMethod
    pitch_phone_averaging: bool
    energy_phone_averaging: bool
    value_separator: str
    save_dir: DirectoryPath
    audio: AudioConfig
    source_data: List[Dataset]

    @validator("save_dir")
    def create_dir(cls, v: str):
        return v

    @convert_paths(kwargs_to_convert=["save_dir"])
    def __init__(self, **data) -> None:
        """Custom init to process file paths"""
        if not data["save_dir"].exists():
            logger.info(f"Directory at {data['save_dir']} does not exist. Creating...")
            data["save_dir"].mkdir(parents=True, exist_ok=True)
        super().__init__(**data, expandable=["audio", "source_data"])
