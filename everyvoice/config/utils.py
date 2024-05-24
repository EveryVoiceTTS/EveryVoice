from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from loguru import logger
from pydantic import PlainSerializer, WithJsonSchema
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated

from everyvoice.utils import load_config_from_json_or_yaml_path

from .validation_helpers import (
    callable_to_string,
    directory_path_must_exist,
    path_is_a_directory,
    relative_to_absolute_path,
    string_to_callable,
)


def load_partials(
    pre_validated_model_dict: dict[Any, Any],
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


PossiblySerializedCallable = Annotated[
    Callable,
    BeforeValidator(string_to_callable),
    PlainSerializer(callable_to_string, return_type=str),
    WithJsonSchema({"type": "string"}, mode="serialization"),  # noqa: F821
    WithJsonSchema({"type": "string"}, mode="validation"),  # noqa: F821
]

PossiblyRelativePath = Annotated[Path, BeforeValidator(relative_to_absolute_path)]

# [Ordering of validators within Annotated](https://docs.pydantic.dev/latest/concepts/validators/#ordering-of-validators-within-annotated)
# Order of validation metadata within Annotated matters. Validation goes from
# right to left and back. That is, it goes from right to left running all
# "before" validators (or calling into "wrap" validators), then left to right
# back out calling all "after" validators.
PossiblyRelativePathMustExist = Annotated[
    Path,
    BeforeValidator(path_is_a_directory),
    BeforeValidator(directory_path_must_exist),
    BeforeValidator(relative_to_absolute_path),
]
