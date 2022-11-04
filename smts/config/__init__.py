from pathlib import Path
from typing import Any, Dict

from smts.config.base_config import SMTSConfig
from smts.config.base_config import __file__ as smts_file

CONFIGS: Dict[str, Any] = {
    "base": SMTSConfig.load_config_from_path(
        Path(smts_file).parent / "base" / "base.yaml"
    ),
    "lj": SMTSConfig.load_config_from_path(Path(smts_file).parent / "lj" / "lj.yaml"),
    "istft": SMTSConfig.load_config_from_path(
        Path(smts_file).parent / "lj" / "lj_istft.yaml"
    ),
    "openslr": SMTSConfig.load_config_from_path(
        Path(smts_file).parent / "openslr" / "openslr.yaml"
    ),
}


class ConfigError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
