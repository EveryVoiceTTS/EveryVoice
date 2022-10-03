from typing import Any, Dict

from smts.config.base_config import BaseConfig
from smts.config.lj_config import BaseConfig as LJConfig

CONFIGS: Dict[str, Any] = {"base": BaseConfig(), "lj": LJConfig()}


class ConfigError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
