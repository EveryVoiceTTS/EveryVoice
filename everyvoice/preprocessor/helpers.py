import tempfile
from multiprocessing import managers
from pathlib import Path

import torch
import torchaudio


def load_audio(audio_path: str | Path) -> tuple[torch.Tensor, int, float]:
    """
    Load an audio file and calculate its duration.

    Args:
        audio_path: path to audio file

    Returns: (audio as a Tensor, sampling rate, duration in seconds)
    """
    audio, sr = torchaudio.load(str(audio_path))
    seconds = len(audio[0]) / sr
    return audio, sr, seconds


def save_tensor(tensor: torch.Tensor, path: str | Path):
    """Create hierarchy before saving a tensor."""
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def save_wav(
    audio: torch.Tensor, path: str | Path, sr: int | None, bits_per_sample: int
):
    """Create hierarchy before saving a wav."""
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(
        str(path),
        audio,
        sr,
        encoding="PCM_S",
        bits_per_sample=bits_per_sample,
    )


class Scaler:
    def __init__(self):
        self._data = []
        self._tensor_data = None
        self.min = None
        self.max = None
        self.std = None
        self.mean = None
        self.norm_min = None
        self.norm_max = None

    def __len__(self):
        return len(self.data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        raise ValueError(
            f"Sorry, you tried to change the data to {value} but it cannot be changed directly. Either Scaler.append(data), or Scaler.clear_data()"
        )

    def append(self, value):
        self._data.append(value)

    def clear_data(self):
        """Clear data"""
        self.__init__()

    def normalize(self, data):
        """Remove mean and normalize to unit variance"""
        return (data - self.mean) / self.std

    def denormalize(self, data):
        """Get de-normalized value"""
        return (data * self.std) + self.mean

    def calculate_stats(self):
        if not len(self):
            return
        if self._tensor_data is None:
            self._tensor_data = torch.cat(self.data)
        non_nan_data = self._tensor_data[~torch.isnan(self._tensor_data)]
        self.min = torch.min(non_nan_data)
        self.max = torch.max(non_nan_data)
        self.mean = torch.nanmean(self._tensor_data)
        self.std = torch.std(non_nan_data)
        self.norm_max = self.normalize(self.max)
        self.norm_min = self.normalize(self.min)
        return {
            "sample_size": len(self),
            "norm_min": float(self.norm_min),
            "norm_max": float(self.norm_max),
            "min": float(self.min),
            "max": float(self.max),
            "mean": float(self.mean),
            "std": float(self.std),
        }


class Counters:
    def __init__(self, manager: managers.SyncManager):
        self._lock = manager.Lock()
        self._processed_files = manager.Value("l", 0)
        self._previously_processed_files = manager.Value("l", 0)
        self._duration = manager.Value("d", 0)
        self._nans = manager.Value("l", 0)
        self._audio_empty = manager.Value("l", 0)
        self._audio_too_long = manager.Value("l", 0)
        self._audio_too_short = manager.Value("l", 0)
        self._multichannel_files = manager.Value("l", 0)
        self._skipped_processes = manager.Value("l", 0)
        self._missing_files = manager.Value("l", 0)
        self._sox_error = manager.Value("l", 0)

    def increment(self, counter: str, increment: int | float = 1):
        with self._lock:
            self.__getattribute__("_" + counter).value += increment

    def value(self, counter):
        with self._lock:
            return self.__getattribute__("_" + counter).value


class SoxError(Exception):
    pass


def apply_sox_effects_to_file(
    infile: str | Path, outfile: str | Path, effects: list[list[str]]
):
    """Use the `sox` command line utility to apply effects to infile, writing to outfile.

    Warning: overwrites outfile if it already exists.
    Warning: the extension given to outfile is significant, as sox will convert to the
             converting file type, so make sure your expension is ".wav" unless you
             really mean to use a different format

    raises: SoxError in case of problem
    """
    import subprocess

    # sox actually takes a flattened list of effects each followed by its arguments
    command_line = sum([["sox", str(infile), str(outfile)], *effects], start=[])
    # print(command_line)

    try:
        result = subprocess.run(command_line, capture_output=True)
    except FileNotFoundError as e:
        raise Exception("Please make sure the sox executable is on your PATH") from e

    # print(result.returncode, len(result.stdout), result.stderr)
    if result.returncode != 0:
        error_log = result.stderr.decode()
        if result.returncode == 1:
            raise SoxError(f"Error in list of sox effects: {error_log}")
        else:
            raise SoxError(f"Error applying sox effects to '{infile}': {error_log}")


def apply_sox_effects_to_tensor(
    infile: str | Path, effects: list[list[str]]
) -> tuple[torch.Tensor, int, float]:
    """Use the `sox` command line utility to apply effects to infile, returning results as a Tensor

    Returns: (audio as a Tensor, result sampling rate, result duration in seconds)
    """
    try:
        outfile = tempfile.NamedTemporaryFile(
            delete=False, prefix="sox_input_", suffix=".wav"
        )
        outfile_name = outfile.name
        outfile.close()

        apply_sox_effects_to_file(infile, outfile_name, effects)
        return load_audio(outfile_name)

    finally:
        Path(outfile_name).unlink()
