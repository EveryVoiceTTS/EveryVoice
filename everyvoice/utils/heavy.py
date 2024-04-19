import random
from typing import Tuple

import numpy as np
import torch
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence

from everyvoice.config.preprocessing_config import AudioSpecTypeEnum
from everyvoice.utils import _flatten


def expand(values, durations):
    out = []
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    if isinstance(values, list):
        return np.array(out)
    elif isinstance(values, torch.Tensor):
        return torch.stack(out)
    elif isinstance(values, np.ndarray):
        return np.array(out)


def collate_fn(data):
    # list-of-dict -> dict-of-lists
    # (see https://stackoverflow.com/a/33046935)
    data = [_flatten(x) for x in data]
    data = {k: [dic[k] for dic in data] for k in data[0]}
    for key in data:
        if isinstance(data[key][0], np.ndarray):
            data[key] = [torch.tensor(x) for x in data[key]]
        if torch.is_tensor(data[key][0]):
            data[key] = pad_sequence(data[key], batch_first=True, padding_value=0)
        if isinstance(data[key][0], int):
            data[key] = torch.IntTensor(data[key])
    return data


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def get_spectral_transform(
    spec_type,
    n_fft,
    win_length,
    hop_length,
    sample_rate=None,
    n_mels=None,
    f_min=0,
    f_max=8000,
):
    if spec_type == AudioSpecTypeEnum.mel.value:
        return T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            norm="slaney",
            center=True,
        )
    elif spec_type == AudioSpecTypeEnum.mel_librosa.value:
        from librosa.filters import mel as librosa_mel

        transform = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad=0,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        mel_basis = librosa_mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
        )
        mel_basis = torch.from_numpy(mel_basis).float()

        def mel_transform(x):
            transform.to(x.device)
            spec = transform(x)
            sine_windowed_spec = torch.sqrt(spec + 1e-9)
            mel = torch.matmul(mel_basis.to(x.device), sine_windowed_spec)
            return mel

        return mel_transform
    elif spec_type == AudioSpecTypeEnum.linear.value:
        return T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
    elif spec_type == AudioSpecTypeEnum.raw.value:
        return T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=None,
        )
    elif spec_type == "istft":
        return T.InverseSpectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length
        )
    else:
        return None


def get_segments(
    t: torch.Tensor, segment_size: int, start=None
) -> Tuple[torch.Tensor, int]:
    """Randomly select a segment from a tensor, if the segment is too short, pad it with zeros

    Args:
        t (torch.Tensor): A tensor (time as second dimension)
        segment_size (int): segment size (should be in frames if spectrogram input or samples if waveform input)
        start (_type_, optional): start at specific input, otherwise random. Defaults to None.

    Returns:
        Tuple[torch.Tensor, int]: the segment plus the start index of the segment
    """
    t_len = t.size(1)
    if t_len >= segment_size:
        max_start = t_len - segment_size - 1
        if start is not None:
            assert (
                start <= max_start
            ), f"Segment start was set to be {start} but max is {max_start}"
        else:
            start = random.randint(0, max_start)
        t = t[:, start : start + segment_size]
    else:
        start = 0
        t = torch.nn.functional.pad(t, (0, segment_size - t_len), "constant")
    return t, start


def get_device_from_accelerator(accelerator: str) -> torch.device:
    """Given an accelerator name ("auto", "cpu", "gpu", "mps"), return it's torch.device equivalent."""
    match accelerator:
        case "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        case "gpu":
            device = torch.device("cuda:0")
        case "cpu" | "mps":
            device = torch.device(accelerator)
        case _:
            device = torch.device("cpu")

    return device
