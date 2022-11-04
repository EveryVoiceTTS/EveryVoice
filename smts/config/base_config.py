import math
from pathlib import Path

from pydantic import root_validator

from smts.config.preprocessing_config import PreprocessingConfig
from smts.config.shared_types import PartialConfigModel
from smts.config.text_config import TextConfig
from smts.model.aligner.DeepForcedAligner.dfaligner.config import (
    DFAlignerModelConfig,
    DFAlignerTrainingConfig,
)
from smts.model.feature_prediction.FastSpeech2_lightning.fs2.config import (
    FastSpeech2ModelConfig,
    FastSpeech2TrainingConfig,
)
from smts.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import (
    HiFiGANModelConfig,
    HiFiGANTrainingConfig,
)
from smts.utils import load_config_from_json_or_yaml_path


class VocoderConfig(PartialConfigModel):
    model: HiFiGANModelConfig
    training: HiFiGANTrainingConfig
    preprocessing: PreprocessingConfig

    @root_validator
    def check_upsample_rate_consistency(cls, values):
        # helper variables
        preprocessing_config: PreprocessingConfig = values["preprocessing"]
        model_config: HiFiGANModelConfig = values["model"]
        sampling_rate = preprocessing_config.audio.input_sampling_rate
        upsampled_sampling_rate = preprocessing_config.audio.output_sampling_rate
        upsample_rate = upsampled_sampling_rate // sampling_rate
        upsampled_hop_size = upsample_rate * preprocessing_config.audio.fft_hop_frames
        upsample_rate_product = math.prod(model_config.upsample_rates)
        # check that same number of kernels and kernel sizes exist
        if len(model_config.upsample_kernel_sizes) != len(model_config.upsample_rates):
            raise ValueError(
                "Number of upsample kernel sizes must match number of upsample rates"
            )
        # Check that kernel sizes are not less than upsample rates and are evenly divisible
        for kernel_size, upsample_rate in zip(
            model_config.upsample_kernel_sizes, model_config.upsample_rates
        ):
            if kernel_size < upsample_rate:
                raise ValueError(
                    f"Upsample kernel size: {kernel_size} must be greater than upsample rate: {upsample_rate}"
                )
            if kernel_size % upsample_rate != 0:
                raise ValueError(
                    f"Upsample kernel size: {kernel_size} must be evenly divisible by upsample rate: {upsample_rate}"
                )
        # check that upsample rate is even multiple of target sampling rate
        if upsampled_sampling_rate % sampling_rate != 0:
            raise ValueError(
                f"Target sampling rate: {upsampled_sampling_rate} must be an even multiple of input sampling rate: {sampling_rate}"
            )
        # check that the upsampling hop size is equal to product of upsample rates
        if model_config.istft_layer:
            upsampled_hop_size /= 4  # istft upsamples the rest
        # check that upsampled hop size is equal to product of upsampling rates
        if upsampled_hop_size != upsample_rate_product:
            raise ValueError(
                f"Upsampled hop size: {upsampled_hop_size} must be equal to product of upsample rates: {upsample_rate_product}"
            )
        # check that the segment size is divisible by product of upsample rates
        if preprocessing_config.audio.vocoder_segment_size % upsample_rate_product != 0:
            raise ValueError(
                f"Vocoder segment size: {preprocessing_config.audio.vocoder_segment_size} must be divisible by product of upsample rates: {upsample_rate_product}"
            )

        return values


class FeaturePredictionConfig(PartialConfigModel):
    model: FastSpeech2ModelConfig
    training: FastSpeech2TrainingConfig
    preprocessing: PreprocessingConfig
    text: TextConfig


class AlignerConfig(PartialConfigModel):
    model: DFAlignerModelConfig
    training: DFAlignerTrainingConfig
    preprocessing: PreprocessingConfig
    text: TextConfig

    @staticmethod
    def load_config_from_path(path: Path) -> "AlignerConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        return AlignerConfig(**config)


class SMTSConfig(PartialConfigModel):
    aligner: AlignerConfig
    feature_prediction: FeaturePredictionConfig
    vocoder: VocoderConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def load_config_from_path(
        path: Path = (Path(__file__).parent / "base" / "base_composed.yaml"),
    ) -> "SMTSConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        return SMTSConfig(**config)
