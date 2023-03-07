from importlib import import_module
from typing import Callable, List, Union

from everyvoice.utils import rel_path_to_abs_path


def convert_callables(*args, kwargs_to_convert: List[str] = []):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            converted_kwargs = {
                k: string_to_callable(v)
                for k, v in kwargs.items()
                if k in kwargs_to_convert
            }
            kwargs = kwargs | converted_kwargs
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def convert_paths(*args, kwargs_to_convert: List[str] = []):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            converted_kwargs = {
                k: rel_path_to_abs_path(v)
                for k, v in kwargs.items()
                if k in kwargs_to_convert
            }
            kwargs = kwargs | converted_kwargs
            return fn(*args, **kwargs)

        return wrapper

    return decorator


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
