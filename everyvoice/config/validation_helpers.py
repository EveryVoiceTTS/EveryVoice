from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger
from pydantic import ValidationInfo


def callable_to_string(function: Callable) -> str:
    """Serialize a Callable to a string-formatted Callable"""
    return ".".join([function.__module__, function.__name__])


def string_to_callable(string: str | Callable) -> Callable:
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


def relative_to_absolute_path(
    value: Any, info: Optional[ValidationInfo] = None
) -> Path | None:
    """
    Helper function to annotate a type.
    This function processes relative paths and either resolve them to absolute
    paths or resolve them with respect to the configuration file they came
    from.
    """
    if value is None:
        return value

    try:
        # Make sure value is a path because it can be a string when we load a
        # model that is not partial.
        path = Path(value)
        if (
            not path.is_absolute()
            and info
            and info.context
            and (config_path := info.context.get("config_path", None))
        ):
            path = (config_path.parent / path).resolve()
        return path
    except TypeError as e:
        # Pydantic needs ValueErrors to raise its ValidationErrors
        raise ValueError from e


def directory_path_must_exist(
    value: Any, info: Optional[ValidationInfo] = None
) -> Path | None:
    """
    Helper function to annotate a type.
    Creates a directory if it doesn't exist.
    """
    assert isinstance(value, Path)
    if (
        info
        and info.context
        and (writing_config := info.context.get("writing_config", None))
    ):
        # We are writing the original config and must temporarily resolve the path.
        (writing_config.resolve() / value).mkdir(parents=True, exist_ok=True)
    else:
        if not value.exists():
            logger.info(f"Directory at {value} does not exist. Creating...")
            value.mkdir(parents=True, exist_ok=True)

    return value


def path_is_a_directory(
    value: Any, info: Optional[ValidationInfo] = None
) -> Path | None:
    """
    Helper function to annotate a type.
    Verifies ala `PathType("dir")` that `value` is a directory.
    """
    if (
        info
        and info.context
        and (writing_config := info.context.get("writing_config", None))
    ):
        # We are writing the original config and must temporarily resolve the path.
        tmp_path = writing_config.resolve() / value
        if not tmp_path.is_dir():
            raise ValueError(f"{tmp_path} is not a directory")
    else:
        try:
            # Make sure value is a path because it can be a string when we load a model that is not partial.
            path = Path(value)
            if not path.is_dir():
                raise ValueError(f"{path} is not a directory")
        except TypeError as e:
            # Pydantic needs ValueErrors to raise its ValidationErrors
            raise ValueError from e

    return value
