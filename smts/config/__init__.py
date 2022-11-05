from pathlib import Path
from typing import Dict

from smts.config.base_config import __file__ as smts_file

CONFIGS: Dict[str, Path] = {
    "base": Path(smts_file).parent / "base" / "base.yaml",
    "lj": Path(smts_file).parent / "lj" / "lj.yaml",
    "istft": Path(smts_file).parent / "lj" / "lj_istft.yaml",
    "openslr": Path(smts_file).parent / "openslr" / "openslr.yaml",
}


class ConfigError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
