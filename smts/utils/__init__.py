import csv
import re
from os.path import dirname, isabs, isfile, splitext
from pathlib import Path
from unicodedata import normalize

import matplotlib.pylab as plt
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from librosa.filters import mel as librosa_mel
from pympi.Praat import TextGrid

import smts

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def rel_path_to_abs_path(path: str, base_path: str = dirname(smts.__file__)):
    if isabs(path):
        return Path(path)
    base_path = Path(base_path)  # type: ignore
    path = Path(path)  # type: ignore
    return (base_path / path).resolve()  # type: ignore


def original_hifigan_leaky_relu(x):
    return F.leaky_relu(x, 0.1)


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def write_filelist(files, path):
    with open(path, "w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=files[0].keys(),
            delimiter="|",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        writer.writeheader()
        for f in files:
            writer.writerow(f)


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


def lower(text):
    return text.lower()


def nfc_normalize(text):
    return normalize("NFC", text)


def load_lj_metadata_hifigan(path):
    with open(
        path,
        "r",
        newline="",
        encoding="utf8",
    ) as f:
        reader = csv.DictReader(
            f,
            fieldnames=["basename", "raw_text", "norm_text"],
            delimiter="|",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        files = list(reader)
    return files


def generic_dict_loader(path, fieldnames=None):
    with open(
        path,
        "r",
        newline="",
        encoding="utf8",
    ) as f:
        reader = csv.DictReader(
            f,
            fieldnames=fieldnames,
            delimiter="|",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        files = list(reader)
    return files


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def read_textgrid(textgrid_path: str):
    check_file_exists(textgrid_path)
    return TextGrid(textgrid_path)


def read_filelist(
    filelist_path: str,
    filename_col: int = 0,
    filename_suffix: str = "",
    text_col: int = 1,
    delimiter: str = "|",
    speaker_col=None,
    language_col=None,
):
    check_file_exists(filelist_path)
    data = []
    with open(filelist_path, encoding="utf8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for line in reader:
            fn, _ = splitext(line[filename_col])
            entry = {"text": line[text_col], "filename": fn + filename_suffix}
            if speaker_col:
                entry["speaker"] = line[speaker_col]
            if language_col:
                entry["language"] = line[language_col]
            data.append(entry)
    return data


def check_file_exists(path: str):
    if not isfile(path):
        raise FileNotFoundError(f"File at {path} could not be found")
