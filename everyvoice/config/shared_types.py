import contextlib
from collections.abc import Mapping
from pathlib import Path
from types import FunctionType
from typing import Callable, Tuple, Union

from loguru import logger
from pydantic import BaseModel, DirectoryPath, Extra, Field, FilePath

from everyvoice.config.utils import convert_callables, convert_paths
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
    def combine_configs(orig_dict: dict, new_dict: dict):
        """See https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"""
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
        # if expandable keys are provided, only expand those
        if "expandable" in data and data["expandable"]:
            for k, v in data.items():
                if k in data["expandable"] and isinstance(v, str):
                    data_to_expand[k] = v
                else:
                    config[k] = v
        else:
            data_to_expand = data
        # remove expandable key
        if "expandable" in config:
            del config["expandable"]
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
    save_dir: DirectoryPath = Path("./logs")
    sub_dir: str = "everyvoice.utils.get_current_time"
    version: str = "base"

    @convert_callables(kwargs_to_convert=["sub_dir"])
    @convert_paths(kwargs_to_convert=["save_dir"])
    def __init__(self, **data) -> None:
        """Custom init to process file paths"""
        # Supress keyerrors because defaults will be used if not supplied
        with contextlib.suppress(KeyError):
            if callable(data["sub_dir"]):
                data["sub_dir"] = data["sub_dir"]()
            if not data["save_dir"].exists():
                logger.info(
                    f"Directory at {data['save_dir']} does not exist. Creating..."
                )
                data["save_dir"].mkdir(parents=True, exist_ok=True)
        super().__init__(**data)


class BaseTrainingConfig(ConfigModel):
    batch_size: int = 16
    train_split: float = 0.9
    save_top_k_ckpts: int = 5
    ckpt_steps: Union[int, None] = None
    ckpt_epochs: Union[int, None] = 1
    max_epochs: int = 1000
    seed: int = 1234
    finetune_checkpoint: Union[FilePath, None] = None
    filelist: Union[Path, FilePath] = Path("./path/to/your/preprocessed/filelist.psv")
    filelist_loader: Callable = generic_dict_loader
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    val_data_workers: int = 0
    train_data_workers: int = 4

    @convert_callables(kwargs_to_convert=["filelist_loader"])
    @convert_paths(kwargs_to_convert=["finetune_checkpoint", "filelist"])
    def __init__(
        self,
        **data,
    ) -> None:
        """Custom init to process file paths"""
        # Supress keyerrors because defaults will be used if not supplied
        super().__init__(
            **data,
        )


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
