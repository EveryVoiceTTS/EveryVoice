from importlib import import_module
from typing import Callable, Union


def string_to_callable(string: Union[str, Callable]) -> Callable:
    """Convert a string to a callable"""
    if callable(string):
        return string
    elif not isinstance(string, str):
        raise ValueError(f"Expected a string or callable, got {type(string)}")
    if "." not in string:
        raise ValueError("String must be in the format <module>.<function>")
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
            f"Cannot find method {function} in module {module}"
        ) from exc
    return function
