import csv
import json
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable

import yaml


def read_unknown_tabular_filelist(
    path,
    delimiter=",",
    quoting=csv.QUOTE_NONE,
    escapechar="\\",
    record_limit: int = 0,  # if non-zero, read only this many records
) -> list[list[str]]:
    """Returns a list of list of cell values instead of a list of dicts as with
        the standard everyvoice.utils.generic_*sv_filelist_reader

    Returns:
        list[list[str]]: a list of rows containing a list of cell values
    """
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


def write_dict_to_config(config: Dict, path: Path):
    """Given an object, write it to file.
       We have to serialize the json first to make use of the custom serializers,
       and we don't always write the json directly since we might want to add extra
       information after serialization.

    Args:
        config (Dict): The Configuration Dict to write
        path (Path): The output path; must end with either json or yaml
    """
    with open(path, "w", encoding="utf8") as f:
        path_string = str(path)
        if path_string.endswith("yaml"):
            yaml.dump(config, f, default_flow_style=None, allow_unicode=True)
        else:
            json.dump(config, f, ensure_ascii=False)
