from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

from loguru import logger
from pydantic import PlainSerializer, WithJsonSchema
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated

from everyvoice.utils import load_config_from_json_or_yaml_path, rel_path_to_abs_path


def load_partials(
    pre_validated_model_dict: Dict[Any, Any],
    partial_keys: Sequence[str],
    config_path: Optional[Path] = None,
):
    """Loads all partials based on a list of partial keys. For this to work,
    your model must have a {key}_config_file: Optional[FilePath] = None field
    defined, and you must have a model_validator(mode="before") that runs this
    function.
    """
    # If there's nothing there, just return the dict
    if not pre_validated_model_dict:
        return pre_validated_model_dict
    # Otherwise go through each key and load it in
    for key in partial_keys:
        key_for_path_to_partial = f"path_to_{key}_config_file"  # I added path_to_ because model_* is a restricted namespace on pydantic models
        if (
            key_for_path_to_partial in pre_validated_model_dict
            and pre_validated_model_dict[key_for_path_to_partial]
        ):
            subconfig_path = Path(pre_validated_model_dict[key_for_path_to_partial])
            if not subconfig_path.is_absolute() and config_path is not None:
                subconfig_path = (config_path.parent / subconfig_path).resolve()
            pre_validated_model_dict[key_for_path_to_partial] = subconfig_path
            # anything defined in the key will override the path
            # so audio would override any values in path_to_audio_config_file
            if key in pre_validated_model_dict:
                logger.info(
                    f"You have both the key {key} and {key_for_path_to_partial} defined in your configuration. We will override values from {key_for_path_to_partial} with values from {key}"
                )
                if isinstance(pre_validated_model_dict[key], dict):
                    pre_validated_model_dict[key] = {
                        **load_config_from_json_or_yaml_path(subconfig_path),
                        **pre_validated_model_dict[key],
                    }
                else:
                    try:
                        # Maybe a model was passed
                        pre_validated_model_dict[key] = {
                            **load_config_from_json_or_yaml_path(subconfig_path),
                            **pre_validated_model_dict[key].model_dump(),
                        }
                    except AttributeError:
                        # If model_dump() doesn't exist then don't try and merge anything just continue and let the model raise a pydantic.ValidationError
                        pass
            else:
                pre_validated_model_dict[key] = load_config_from_json_or_yaml_path(
                    subconfig_path
                )
    return pre_validated_model_dict


def callable_to_string(function: Callable) -> str:
    """Serialize a Callable to a string-formatted Callable"""
    return ".".join([function.__module__, function.__name__])


def string_to_callable(string: Union[str, Callable]) -> Callable:
    """De-serialize a string-formatted Callable to a Callable"""
    if callable(string):
        return string
    elif not isinstance(string, str):
        raise ValueError(f"Expected a string or callable, got {type(string)}")
    if "." not in string:
        # Just return a function that returns the string if
        # it's not in the <module>.<function> format
        def curried(*argv, **kwargs):
            return string

        return curried
    module_name, function_name = string.rsplit(".", 1)
    try:
        module = import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module {module_name} - this must be a valid module"
        ) from e
    try:
        function = getattr(module, function_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Cannot find method {function_name} in module {module}"
        ) from exc
    return function


PossiblySerializedCallable = Annotated[
    Callable,
    BeforeValidator(string_to_callable),
    PlainSerializer(callable_to_string, return_type=str),
    WithJsonSchema({"type": "string"}, mode="serialization"),  # noqa: F821
    WithJsonSchema({"type": "string"}, mode="validation"),  # noqa: F821
]
PossiblyRelativePath = Annotated[Path, BeforeValidator(rel_path_to_abs_path)]