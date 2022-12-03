from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Tuple, Union

from loguru import logger
from pydantic import BaseModel, DirectoryPath, Extra, FilePath

from smts.config.utils import convert_callables, convert_paths
from smts.utils import load_config_from_json_or_yaml_path, rel_path_to_abs_path


class ConfigModel(BaseModel):
    class Config:
        extra = Extra.forbid
        use_enum_values = True

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
    name: str
    save_dir: DirectoryPath
    sub_dir: str
    version: str

    @convert_callables(kwargs_to_convert=["sub_dir"])
    @convert_paths(kwargs_to_convert=["save_dir"])
    def __init__(self, **data) -> None:
        """Custom init to process file paths"""
        if callable(data["sub_dir"]):
            data["sub_dir"] = data["sub_dir"]()
        if not data["save_dir"].exists():
            logger.info(f"Directory at {data['save_dir']} does not exist. Creating...")
            data["save_dir"].mkdir(parents=True, exist_ok=True)
        super().__init__(**data)


class BaseTrainingConfig(ConfigModel):
    batch_size: int
    train_split: float
    save_top_k_ckpts: int
    ckpt_steps: Union[int, None]
    ckpt_epochs: Union[int, None]
    max_epochs: int
    seed: int
    finetune_checkpoint: Union[FilePath, None]
    filelist: Union[Path, FilePath]
    filelist_loader: Callable
    logger: LoggerConfig
    val_data_workers: int
    train_data_workers: int

    @convert_callables(kwargs_to_convert=["filelist_loader"])
    @convert_paths(kwargs_to_convert=["finetune_checkpoint", "filelist"])
    def __init__(
        self,
        **data,
    ) -> None:
        """Custom init to process file paths"""
        if not data["filelist"].exists():
            logger.warning(
                f"Filelist {data['filelist']} does not exist. If you're just preprocessing, that's fine, otherwise this will cause an error"
            )
        super().__init__(
            **data,
        )


class BaseOptimizer(ConfigModel):
    learning_rate: float
    eps: float
    weight_decay: int


class RMSOptimizer(BaseOptimizer):
    alpha: float
    name: str = "rms"


class AdamOptimizer(BaseOptimizer):
    betas: Tuple[float, float]
    name: str = "adam"


class AdamWOptimizer(BaseOptimizer):
    betas: Tuple[float, float]
    name: str = "adamw"


class NoamOptimizer(AdamOptimizer):
    warmup_steps: int
    name: str = "noam"
