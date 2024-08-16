from os import PathLike
from typing import Any, BinaryIO, Union


def load_squim_objective_model() -> tuple[Any, int]:
    """Load the objective Squim Model. See https://pytorch.org/audio/main/tutorials/squim_tutorial.html

    Returns:
        tuple[Any, int]: a tuple containing the model and the required sampling rate
    """
    from torchaudio.pipelines import SQUIM_OBJECTIVE

    model = SQUIM_OBJECTIVE.get_model()
    model_sampling_rate = 16000
    return (model, model_sampling_rate)


def load_squim_subjective_model() -> tuple[Any, int]:
    """Load the subjective Squim Model. See https://pytorch.org/audio/main/tutorials/squim_tutorial.html

    Returns:
        tuple[Any, int]: a tuple containing the model and the required sampling rate
    """
    from torchaudio.pipelines import SQUIM_SUBJECTIVE

    model = SQUIM_SUBJECTIVE.get_model()
    model_sampling_rate = 16000
    return (model, model_sampling_rate)


def process_audio(path: Union[BinaryIO, str, PathLike], sampling_rate: int):
    import torchaudio

    audio, sr = torchaudio.load(str(path))
    # Must be 16 kHz
    if sr != sampling_rate:
        audio = torchaudio.functional.resample(audio, sr, sampling_rate)
    # Must have channel dimension
    if len(audio.size()) < 2:
        audio = audio.unsqueeze(0)
    # Must be mono audio
    if audio.size(0) != 1:
        raise ValueError("Audio for evaluation must be mono (single channel)")
    return audio


def calculate_objective_metrics_from_single_path(
    audio_path, model, model_sampling_rate
) -> tuple[float, float, float]:
    import torch

    audio = process_audio(audio_path, model_sampling_rate)
    with torch.no_grad():
        stoi_hyp, pesq_hyp, si_sdr_hyp = model(audio)
    return float(stoi_hyp), float(pesq_hyp), float(si_sdr_hyp)


def calculate_subjective_metrics_from_single_path(
    audio_path, non_matching_reference_path, model, model_sampling_rate
) -> float:
    import torch

    audio = process_audio(audio_path, model_sampling_rate)
    nmr_audio = process_audio(non_matching_reference_path, model_sampling_rate)
    with torch.no_grad():
        mos = model(audio, nmr_audio)
    return float(mos)
