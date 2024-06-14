import csv
import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from unicodedata import normalize

import yaml
from loguru import logger

from everyvoice import exceptions
from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel

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


def filter_dataset_based_on_target_text_representation_level(
    target_text_representation_level: TargetTrainingTextRepresentationLevel,
    dataset: list[dict],
    name: str,
    batch_size: int,
) -> list[dict]:
    # remove dataset samples that don't exist for the target training representation
    match target_text_representation_level:
        case TargetTrainingTextRepresentationLevel.characters:
            target_training_text_key = "character_tokens"
        case (  # noqa: E211
            TargetTrainingTextRepresentationLevel.ipa_phones
            | TargetTrainingTextRepresentationLevel.phonological_features
        ):
            target_training_text_key = "phone_tokens"
        case _:
            raise NotImplementedError(
                f"{target_text_representation_level} have not yet been implemented."
            )

    filtered_dataset = [
        item
        for item in dataset
        if target_training_text_key in item and item[target_training_text_key]
    ]
    n_filtered_items = len(dataset) - len(filtered_dataset)
    if n_filtered_items:
        logger.warning(
            f"Removing {n_filtered_items} from your {name} set because they do not have text values for the target training representation level {target_text_representation_level}. Either change the target training representation level or update the information in your data and re-run preprocessing if you want to use this data."
        )
        dataset = filtered_dataset
    if batch_size > len(filtered_dataset):
        logger.error(
            f"Sorry you do not have enough {target_text_representation_level} data in your current {name} filelist to run the model with a batch size of {batch_size}."
        )
        sys.exit(1)
    return dataset


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
        # The base set of expected fieldnames
        fieldnames = [
            "basename",
            "language",
            "speaker",
            "characters",
            "character_tokens",
            "phones",
            "phone_tokens",
        ]
        # The fieldnames we actually found
        found_fieldnames = sorted(files[0].keys())
        # Use fieldnames in the order we expect, and append unexpected ones to the end
        fieldnames = [x for x in fieldnames if x in found_fieldnames] + [
            x for x in found_fieldnames if x not in fieldnames
        ]
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
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


def read_festival(
    path,
    record_limit: int = 0,  # if non-zero, read only this many records
    text_field_name: str = "text",
):
    """Read Festival format into filelist
    Args:
        path (Path): Path to festival format filelist
        record_limit: if non-zero, read only that many records
        text_field_name (str): the keyname for the returned text. Default is 'text'.
    Raises:
        ValueError: the file is not valid festival input
    """
    festival_pattern = re.compile(
        r"""
    \(\s*
    (?P<basename>[\w\d\-\_.]*)
    \s*
    "(?P<text>[^"]*)"
    \s*\)
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
                data.append({"basename": basename, text_field_name: text})
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
    path: str | os.PathLike,
    delimiter="|",
    quoting=csv.QUOTE_NONE,
    escapechar="\\",
    fieldnames=None,
    file_has_header_line=True,
    record_limit: int = 0,
) -> list[dict]:
    """This is the base function that should be used to parse *sv style tabular filelists.
        The psv variant can also be imported with generic_psv_filelist_reader
        The csv variant can be imported with generic_csv_filelist_reader

    Args:
        path (_type_): path to *sv tabular filelist
        delimiter (str, optional): column delimiter. Defaults to "|".
        quoting (_type_, optional): quoting strategy. Defaults to csv.QUOTE_NONE.
        escapechar (str, optional): escape character. Defaults to "\".
        fieldnames (_type_, optional): fieldnames to parse. Defaults to None.
        file_has_header_line (bool, optional): whether file has header line. Defaults to True.
        record_limit (int): if non-zero, read only this many records. Defaults to 0.

    Returns:
        list[dict]: a list of dicts representing the rows in the filelist
    """
    assert fieldnames is not None or file_has_header_line
    f: Iterable[str]
    with open(path, "r", newline="", encoding="utf8") as f:
        if record_limit:
            f = islice(f, record_limit)
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
        files = []
        for file in list(reader):
            if "basename" in file:
                file["basename"] = os.path.splitext(file["basename"])[0]
            files.append(file)
    return files


generic_psv_filelist_reader = generic_dict_loader
generic_xsv_filelist_reader = generic_dict_loader
generic_csv_filelist_reader = partial(generic_dict_loader, delimiter=",")


def collapse_whitespace(text):
    """
    >>> collapse_whitespace("  asdf  	   qwer   ")
    ' asdf qwer '
    """
    return re.sub(_whitespace_re, " ", text)


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
