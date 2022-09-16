import csv
import re
from os.path import isfile, splitext
from unicodedata import normalize

import torch
import torchaudio.transforms as T
from librosa.filters import mel as librosa_mel_fn
from pympi.Praat import TextGrid

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """spectral normalization"""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """spectral denormalization"""
    return torch.exp(x) / C


def write_filelist(self, files, path):
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


class LibrosaMelSpectrogram:
    # TODO: Fix this, or remove it
    def __init__(
        self,
        n_fft,
        n_mels,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        center=False,
    ):
        self.n_fft = n_fft
        self.num_mels = n_mels
        self.sampling_rate = sample_rate
        self.hop_size = hop_length
        self.win_size = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.center = center
        self.mel_basis = {}
        self.hann_window = {}

    def __call__(self, y):
        """From HiFiGAN"""
        if torch.min(y) < -1.0:
            print("min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("max value is ", torch.max(y))

        if self.f_max not in self.mel_basis:
            mel = librosa_mel_fn(
                self.sampling_rate, self.n_fft, self.num_mels, self.f_min, self.f_max
            )
            self.mel_basis[f"{str(self.f_max)}_{str(y.device)}"] = (
                torch.from_numpy(mel).float().to(y.device)
            )

            self.hann_window[str(y.device)] = torch.hann_window(self.win_size).to(
                y.device
            )
        y = torch.nn.functional.pad(
            y,
            (
                int((self.n_fft - self.hop_size) / 2),
                int((self.n_fft - self.hop_size) / 2),
            ),
            mode="reflect",
        )

        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window[str(y.device)],
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
        )

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(self.mel_basis[f"{str(self.f_max)}_{str(y.device)}"], spec)
        spec = dynamic_range_compression_torch(spec)

        return spec


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
    elif spec_type == "librosa":
        return LibrosaMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
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
