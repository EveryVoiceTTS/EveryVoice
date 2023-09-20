from importlib import import_module
from pathlib import Path
from typing import Callable, Union

from pydantic import PlainSerializer, WithJsonSchema
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated

from everyvoice.utils import rel_path_to_abs_path


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
