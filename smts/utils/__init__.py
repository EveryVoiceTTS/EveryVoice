import csv
import re
from os.path import isfile, splitext
from unicodedata import normalize

import matplotlib.pylab as plt
import torch.nn.functional as F
import torchaudio.transforms as T
from pympi.Praat import TextGrid

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")


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
    if spec_type == "mel":
        return T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
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
