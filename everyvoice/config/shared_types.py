from collections.abc import Mapping, Sequence
from pathlib import Path
from types import FunctionType
from typing import Callable, Tuple, Union

from loguru import logger
from pydantic import BaseModel, DirectoryPath, Extra, Field, FilePath, validator
from pydantic.fields import ModelField

from everyvoice.config.utils import string_to_callable
from everyvoice.utils import (
    generic_dict_loader,
    load_config_from_json_or_yaml_path,
    rel_path_to_abs_path,
)


class ConfigModel(BaseModel):
    class Config:
        extra = Extra.forbid
        use_enum_values = True
        json_encoders = {
            Callable: lambda fn: ".".join(
                [fn.__module__, fn.__name__]
            ),  # This doesn't seem to work for some reason: https://github.com/pydantic/pydantic/issues/4151
            FunctionType: lambda fn: ".".join(
                [fn.__module__, fn.__name__]
            ),  # But this does
        }

    def update_config(self, new_config: dict):
        """Update the config with new values"""
        new_data = self.combine_configs(dict(self), new_config)
        self.__init__(**new_data)  # type: ignore
        return self

    @staticmethod
    def combine_configs(orig_dict: Union[dict, Sequence], new_dict: dict):
        """See https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"""
        if isinstance(orig_dict, Sequence):
            orig_list = list(orig_dict)
            for key_s, val in new_dict.items():
                key_i = int(key_s)
                if isinstance(val, Mapping):
                    tmp = ConfigModel.combine_configs(orig_list[key_i], val)  # type: ignore
                    orig_list[key_i] = tmp
                else:
                    orig_list[key_i] = val
            return orig_list

        orig_dict = dict(orig_dict)
        new_dict = dict(new_dict)
        for key, val in new_dict.items():
            if isinstance(val, Mapping):
                tmp = ConfigModel.combine_configs(orig_dict.get(key, {}), val)  # type: ignore
                orig_dict[key] = tmp
            else:
                orig_dict[key] = new_dict[key]
        return orig_dict


class PartialConfigModel(ConfigModel):
    def __init__(self, **data) -> None:
        """Allow for partial configurations"""
        config = {}
        data_to_expand = {}
        # TODO: this is awkward and should be fixed
        expandable_keys = [
            "aligner",
            "audio",
            "feature_prediction",
            "preprocessing",
            "source_data",
            "sox_effects",
            "text",
            "training",
            "vocoder",
        ]
        for k, v in data.items():
            if k in expandable_keys and isinstance(v, str):
                data_to_expand[k] = v
            else:
                config[k] = v
        # first extend any paths
        if "extend_from" in data:
            path = rel_path_to_abs_path(data["extend_from"])
            config = load_config_from_json_or_yaml_path(path)
            data_to_expand = self.combine_configs(config, data["override_with"])

        for key, subconfig in data_to_expand.items():
            if isinstance(subconfig, str):
                subconfig = rel_path_to_abs_path(subconfig)
            if isinstance(subconfig, Path):
                subconfig = load_config_from_json_or_yaml_path(subconfig)
            config[key] = subconfig

        super().__init__(**config)


class LoggerConfig(ConfigModel):
    name: str = "BaseExperiment"
    save_dir: DirectoryPath = Path("./logs_and_checkpoints")
    sub_dir: str = "everyvoice.utils.get_current_time"
    version: str = "base"

    # always=False so that value doesn't get called if using default (i.e. allows config-wizard to work properly)
    @validator("sub_dir", pre=True, always=False)
    def convert_callable_sub_dir(cls, v, values):
        func = string_to_callable(v)
        called = func()
        values["sub_dir"] = called
        return called

    @validator("save_dir", pre=True, always=True)
    def convert_path(cls, v, values):
        path = Path(v)
        values["save_dir"] = path
        if not path.exists():
            logger.info(f"Directory at {path} does not exist. Creating...")
            path.mkdir(parents=True, exist_ok=True)
        return path


class BaseTrainingConfig(ConfigModel):
    batch_size: int = 16
    save_top_k_ckpts: int = 5
    ckpt_steps: Union[int, None] = None
    ckpt_epochs: Union[int, None] = 1
    max_epochs: int = 1000
    max_steps: int = 100000
    finetune_checkpoint: Union[FilePath, None] = None
    training_filelist: Union[Path, FilePath] = Path(
        "./path/to/your/preprocessed/training_filelist.psv"
    )
    validation_filelist: Union[Path, FilePath] = Path(
        "./path/to/your/preprocessed/validation_filelist.psv"
    )
    filelist_loader: Callable = generic_dict_loader
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    val_data_workers: int = 0
    train_data_workers: int = 4

    @validator("filelist_loader", pre=True, always=True)
    def convert_callable_filelist_loader(cls, v, values):
        func = string_to_callable(v)
        values["filelist_loader"] = func
        return func

#    @validator(
#        "finetune_checkpoint",
#        "training_filelist",
#        "validation_filelist",
#        pre=True,
#        always=True,
#    )
#    def convert_paths(cls, v, values, field: ModelField):
#        path = rel_path_to_abs_path(v)
#        values[field.name] = path
#        return path


class BaseOptimizer(ConfigModel):
    learning_rate: float = 1e-4
    eps: float = 1e-8
    weight_decay: float = 0.01


class RMSOptimizer(BaseOptimizer):
    alpha: float = 0.99
    name: str = "rms"


class AdamOptimizer(BaseOptimizer):
    betas: Tuple[float, float] = (0.9, 0.98)
    name: str = "adam"


class AdamWOptimizer(BaseOptimizer):
    betas: Tuple[float, float] = (0.9, 0.98)
    name: str = "adamw"


class NoamOptimizer(AdamOptimizer):
    warmup_steps: int = 4000
    name: str = "noam"
