import csv
import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from itertools import islice
from os.path import splitext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from unicodedata import normalize

import yaml
from loguru import logger
from pydantic import ValidationInfo

from everyvoice import exceptions

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")
# Regular expression matching non-slug characters:
special_chars = re.compile(r"[\W]+")


def slugify(
    text: str, repl: str = "-", limit_to_n_characters: Optional[int] = None
) -> str:
    """Create a slugified representation of input text
       (i.e. without special characters) and optionally
       limited to a specific number of characters.

       >>> slugify('gya?a')
       'gya-a'

       >>> slugify('gya?a', repl='')
       'gyaa'

       >>> slugify('gya?a', limit_to_n_characters=3)
       'gya'

       >>> slugify('!@#$%^&*(){}<>? :;|+=/\\'"', limit_to_n_characters=3)
       '-'

       >>> slugify('!@#$%^&*(){}<>? :;|+=/\\'"', repl='', limit_to_n_characters=3)
       ''

    Args:
        text (str): text to slugify
        repl (str): character to replace special character with. defaults to -
        limit_to_n_characters (Optional[int]): return first n characters of output

    Returns:
        str: slugified string
    """
    slugified_text = re.sub(special_chars, repl, text)

    if limit_to_n_characters is None:
        return slugified_text
    else:
        return slugified_text[:limit_to_n_characters]


def check_dataset_size(batch_size: int, number_of_samples: int, name: str):
    if batch_size > number_of_samples:
        reduce_train_split = (
            "You can also decrease the train split to increase the number of samples in your validation dataset."
            if "val" in name
            else ""
        )
        logger.error(
            f"Your {name} dataset only has {number_of_samples} samples, but you have a defined batch size of {batch_size}. Please either add more data or decrease your batch size. {reduce_train_split}"
        )
        sys.exit(1)


def return_configs_from_dir(dir: Path) -> Dict[str, Path]:
    return {os.path.basename(path)[:-5]: path for path in dir.glob("*.yaml")}


def get_current_time():
    return str(int(datetime.now().timestamp()))


def _flatten(structure, key="", path="", flattened=None):
    """
    >>> _flatten({"a": {"b": 2, "c": {"d": "e"}, "f": 4}, "g": 5})
    {'a_b': 2, 'a_c_d': 'e', 'a_f': 4, 'g': 5}
    """
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
    with open(path, "r", encoding="utf8") as f:
        config = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
    if not config:
        raise exceptions.InvalidConfiguration(f"Your configuration at {path} was empty")
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


def relative_to_absolute_path(
    value: Any, info: Optional[ValidationInfo] = None
) -> Path | None:
    """
    Helper function to annotate a type.
    This function processes relative paths and either resolve them to absolute
    paths or resolve them with respect to the configuration file they came
    from.
    """
    if value is None:
        return value

    try:
        # Make sure value is a path because it can be a string when we load a
        # model that is not partial.
        path = Path(value)
        if (
            not path.is_absolute()
            and info
            and info.context
            and (config_path := info.context.get("config_path", None))
        ):
            path = (config_path.parent / path).resolve()
        return path
    except TypeError as e:
        # Pydantic needs ValueErrors to raise its ValidationErrors
        raise ValueError from e


def directory_path_must_exist(
    value: Any, info: Optional[ValidationInfo] = None
) -> Path | None:
    """
    Helper function to annotate a type.
    Creates a directory if it doesn't exist.
    """
    assert isinstance(value, Path)
    if (
        info
        and info.context
        and (writing_config := info.context.get("writing_config", None))
    ):
        # We are writing the original config and must temporarily resolve the path.
        (writing_config.resolve() / value).mkdir(parents=True, exist_ok=True)
    else:
        if not value.exists():
            logger.info(f"Directory at {value} does not exist. Creating...")
            value.mkdir(parents=True, exist_ok=True)

    return value


def path_is_a_directory(
    value: Any, info: Optional[ValidationInfo] = None
) -> Path | None:
    """
    Helper function to annotate a type.
    Verifies ala `PathType("dir")` that `value` is a directory.
    """
    if (
        info
        and info.context
        and (writing_config := info.context.get("writing_config", None))
    ):
        # We are writing the original config and must temporarily resolve the path.
        tmp_path = writing_config.resolve() / value
        if not tmp_path.is_dir():
            raise ValueError(f"{tmp_path} is not a directory")
    else:
        try:
            # Make sure value is a path because it can be a string when we load a model that is not partial.
            path = Path(value)
            if not path.is_dir():
                raise ValueError(f"{path} is not a directory")
        except TypeError as e:
            # Pydantic needs ValueErrors to raise its ValidationErrors
            raise ValueError from e

    return value


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
    with open(path, "w", encoding="utf8") as f:
        if not files:
            logger.warning(f"Writing empty filelist file {path}")
            print("", file=f)  # header line, empty because we don't know the fields
            return
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
    """
    >>> lower("MiXeD ÇÀSÉ")
    'mixed çàsé'
    """
    return text.lower()


def nfc_normalize(text):
    """
    >>> nfc_normalize("éçà")
    'éçà'
    """
    return normalize("NFC", text)


def load_lj_metadata_hifigan(path):
    with open(path, "r", newline="", encoding="utf8") as f:
        reader = csv.DictReader(
            f,
            fieldnames=["basename", "raw_text", "text"],
            delimiter="|",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        files = list(reader)
    return files


def read_festival(
    path,
    record_limit: int = 0,  # if non-zero, read only this many records
):
    """Read Festival format into filelist
    Args:
        path (Path): Path to festival format filelist
        record_limit: if non-zero, read only that many records
    Raises:
        ValueError: the file is not valid festival input
    """
    festival_pattern = re.compile(
        r"""
    (\s*
    (?P<basename>[\w\d\-\_]*)
    \s*
    "(?P<text>[^"]*)"
     )
    """,
        re.VERBOSE,
    )
    data = []
    f: Iterable[str]
    with open(path, encoding="utf-8") as f:
        if record_limit:
            f = islice(f, record_limit)
        for line in f:
            if match := re.search(festival_pattern, line.strip()):
                basename = match["basename"].strip()
                text = match["text"].strip()
                data.append({"basename": basename, "text": text})
            else:
                raise ValueError(f'File {path} is not in the "festival" format.')
    return data


def sniff_and_return_filelist_data(path):
    """Sniff csv, and return dialect if not festival format:
    ( LJ0002 "this is the festival format" )
    Args:
        path (Path): path to filelist
    Returns:
        False if not csv
    """
    festival_pattern = re.compile(r'\( [\w\d_]* "[^"]*" \)')
    with open(path, newline="", encoding="utf8") as f:
        data = f.read(1024)
        f.seek(0)
        if re.search(festival_pattern, data):
            return read_festival(path)
        else:
            dialect = csv.Sniffer().sniff(data)
            reader = csv.DictReader(f, dialect=dialect)
            return list(reader)


def generic_dict_loader(
    path,
    delimiter="|",
    quoting=csv.QUOTE_NONE,
    escapechar="\\",
    fieldnames=None,
    file_has_header_line=True,
):
    assert fieldnames is not None or file_has_header_line
    with open(path, "r", newline="", encoding="utf8") as f:
        reader = csv.DictReader(
            f,
            fieldnames=fieldnames,
            delimiter=delimiter,
            quoting=quoting,
            escapechar=escapechar,
        )
        # When fieldnames is given, csv.DictReader assumes the first line is a data
        # line.  Skip it if the file has a header line.
        if fieldnames and file_has_header_line:
            next(reader)
        files = list(reader)
    return files


generic_psv_dict_reader = generic_dict_loader


def generic_csv_reader(
    path,
    delimiter=",",
    quoting=csv.QUOTE_NONE,
    escapechar="\\",
    record_limit: int = 0,  # if non-zero, read only this many records
):
    f: Iterable[str]
    with open(path, "r", newline="", encoding="utf8") as f:
        if record_limit:
            f = islice(f, record_limit)
        reader = csv.reader(
            f,
            delimiter=delimiter,
            quoting=quoting,
            escapechar=escapechar,
        )
        files = list(reader)
    return files


def collapse_whitespace(text):
    """
    >>> collapse_whitespace("  asdf  	   qwer   ")
    ' asdf qwer '
    """
    return re.sub(_whitespace_re, " ", text)


def read_filelist(
    filelist_path: Union[str, os.PathLike],
    filename_col: int = 0,
    filename_suffix: str = "",
    text_col: int = 1,
    delimiter: str = "|",
    speaker_col=None,
    language_col=None,
):
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


@contextmanager
def tqdm_joblib_context(tqdm_instance):
    """Context manager to make tqdm compatible with joblib.Parallel

    Runs the parallel jobs using joblib, but displays the nicer tqdm progress bar
    Only tested with tqdm.tqdm, but should also work with tqdm.notepad.tqdm and
    other variants

    Usage:
        with tqdm_joblib_context(tqdm(desc="my description", total=len(job_list))):
            joblib.Parallel(n_jobs=cpus)(delayed(fn)(item) for item in job_list)
    """
    import joblib.parallel

    class ParallelCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, out):
            tqdm_instance.update(n=self.batch_size)
            super().__call__(out)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = ParallelCallback
    try:
        yield
    finally:
        tqdm_instance.close()
        joblib.parallel.BatchCompletionCallBack = old_callback


def n_times(n: int) -> str:
    """Return a grammatically correct version of n times for n > 0.

    >>> n_times(1)
    'once'
    >>> n_times(2)
    'twice'
    >>> n_times(1001)
    '1001 times'
    """
    if n == 1:
        return "once"
    if n == 2:
        return "twice"
    return f"{n} times"
