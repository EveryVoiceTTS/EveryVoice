import csv
import json
import os
import re
from datetime import datetime
from os.path import dirname, isabs, isfile, splitext
from pathlib import Path
from typing import Any, Dict, List, Union
from unicodedata import normalize

import yaml
from loguru import logger
from pympi.Praat import TextGrid

import smts

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")


def return_configs_from_dir(dir: Path) -> Dict[str, Path]:
    return {os.path.basename(path)[:-5]: path for path in dir.glob("*.yaml")}


def get_current_time():
    return str(int(datetime.now().timestamp()))


def _flatten(structure, key="", path="", flattened=None):
    if flattened is None:
        flattened = {}
    if not isinstance(structure, dict):
        flattened[(f"{path}_" if path else "") + key] = structure
    else:
        for new_key, value in structure.items():
            _flatten(value, new_key, (f"{path}_" if path else "") + key, flattened)
    return flattened


def load_config_from_json_or_yaml_path(path: Path):
    if not path.exists():
        raise ValueError(f"Config file '{path}' does not exist")
    with open(path, "r") as f:
        config = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
    return config


def expand_config_string_syntax(config_arg: str) -> dict:
    """Expand a string of the form "key1=value1" into a dict."""
    config_dict: Any = {}
    try:
        key, value = config_arg.split("=")
    except ValueError as e:
        raise ValueError(f"Invalid config string: {config_arg} - missing '='") from e
    current_dict = config_dict
    keys = key.split(".")
    for key in keys[:-1]:
        current_dict[key] = {}
        current_dict = current_dict[key]
    current_dict[keys[-1]] = value
    return config_dict


def update_config_from_cli_args(arg_list: List[str], original_config):
    if arg_list is None or not arg_list:
        return original_config
    for arg in arg_list:
        key, value = arg.split("=")
        logger.info(f"Updating config '{key}' to value '{value}'")
        original_config = original_config.update_config(
            expand_config_string_syntax(arg)
        )
    return original_config


def update_config_from_path(config_path: Path, original_config):
    if config_path is None:
        return original_config
    logger.info(f"Loading and updating config from '{config_path}'")
    config_override = load_config_from_json_or_yaml_path(config_path)
    return original_config.update_config(config_override)


def rel_path_to_abs_path(
    path: Union[None, str], base_path: str = dirname(smts.__file__)
):
    if path is None:
        return None
    if isabs(path):
        return Path(path)
    base_path = Path(base_path)  # type: ignore
    path = Path(path)  # type: ignore
    return (base_path / path).resolve()  # type: ignore


def original_hifigan_leaky_relu(x):
    import torch.nn.functional as F

    return F.leaky_relu(x, 0.1)


def plot_spectrogram(spectrogram):
    import matplotlib.pylab as plt

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
            fieldnames=["basename", "raw_text", "text"],
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


def generic_csv_loader(path):
    with open(path, "r", newline="", encoding="utf8") as f:
        reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        return list(reader)


def write_dict(path, data, fieldnames):
    with open(
        path,
        "w",
        newline="",
        encoding="utf8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="|",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        writer.writeheader()
        for line in data:
            writer.writerow(line)


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
