from pathlib import Path

from loguru import logger


def validate_path(value: str, is_dir=True, is_file=True, exists=True):
    path = Path(value).expanduser()
    if not is_dir and not is_file:
        raise ValueError(
            "The path must be either a file or directory, but both were set to False"
        )
    if not is_file and path.is_file():
        logger.warning(
            f"Sorry, the path at '{path.absolute()}' is a file. Please select a directory."
        )
        return False
    if not is_dir and path.is_dir():
        logger.warning(
            f"Sorry, the path at '{path.absolute()}' is a file. Please select a directory."
        )
        return False
    if not exists and path.exists():
        logger.warning(
            f"Sorry, the path at '{path.absolute()}' already exists. Please choose another path."
        )
        return False
    if exists and not path.exists():
        logger.warning(
            f"Sorry, the path at '{path.absolute()}' doesn't exist. Please choose another path."
        )
        return False
    return True
