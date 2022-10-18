from typing import Any, Dict

from smts.config.base_config import BaseConfig
from smts.config.lj_config import BaseConfig as LJConfig
from smts.config.lj_config_istft import BaseConfig as LJConfigIstft
from smts.config.openslr_config_istft import BaseConfig as OpenSLRConfigIstft
from smts.config.xh_config_istft import BaseConfig as XHConfigIstft

CONFIGS: Dict[str, Any] = {
    "base": BaseConfig(),
    "lj": LJConfig(),
    "istft": LJConfigIstft(),
    "xh": XHConfigIstft(),
    "openslr": OpenSLRConfigIstft(),
}


class ConfigError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
