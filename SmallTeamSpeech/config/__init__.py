from .base_config import BaseConfig

CONFIGS = {"base": BaseConfig()}


class ConfigError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
