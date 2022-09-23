from .base_config import BaseConfig
from .lj_config import BaseConfig as LJConfig

CONFIGS = {"base": BaseConfig(), "lj": LJConfig()}


class ConfigError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
