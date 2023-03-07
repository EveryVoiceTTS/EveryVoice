from pathlib import Path
from typing import Dict

from everyvoice.config import __file__ as everyvoice_file

CONFIGS: Dict[str, Path] = {
    "base": Path(everyvoice_file).parent / "base" / "base_composed.yaml",
    "lj": Path(everyvoice_file).parent / "lj" / "lj.yaml",
    "istft": Path(everyvoice_file).parent / "lj" / "lj_istft.yaml",
    "openslr": Path(everyvoice_file).parent / "openslr" / "openslr.yaml",
}


class ConfigError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
