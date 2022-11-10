import numpy as np
import torch
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence

from smts.utils import _flatten


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
            pad_val = 1 if "silence_mask" in key else 0
            data[key] = pad_sequence(data[key], batch_first=True, padding_value=pad_val)
        if isinstance(data[key][0], int):
            data[key] = torch.tensor(data[key]).long()
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
    if spec_type == "mel-torch":
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
    elif spec_type == "mel-librosa":
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
    elif spec_type == "linear":
        return T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
    elif spec_type == "raw":
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
