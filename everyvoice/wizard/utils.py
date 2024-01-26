import json
from pathlib import Path
from typing import Dict

import yaml


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
