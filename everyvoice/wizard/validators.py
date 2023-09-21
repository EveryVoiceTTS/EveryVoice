from pathlib import Path


def validate_path(value: str, is_dir=False, is_file=False, exists=False):
    path = Path(value).expanduser()
    if bool(is_dir) == bool(is_file):
        raise ValueError(
            "The path must be either a file or directory, but both or neither were specified."
        )
    if is_dir and path.is_file():
        print(f"Sorry, the path '{path}' is a file. Please select a directory.")
        return False
    if is_file and path.is_dir():
        print(f"Sorry, the path '{path}' is a directory. Please select a file.")
        return False
    if not exists and path.exists():
        print(f"Sorry, the path '{path}' already exists. Please choose another path.")
        return False
    if exists and not path.exists():
        print(f"Sorry, the path '{path}' doesn't exist. Please choose another path.")
        return False
    return True
