"""
/home/sam037/.conda/envs/EveryVoice.sl/lib/python3.9/site-packages/pydantic/_internal/_config.py:321: UserWarning: Valid config keys have changed in V2:
* 'validate_all' has been renamed to 'validate_default'
  warnings.warn(message, UserWarning)
Traceback (most recent call last):
  File "/home/sam037/.conda/envs/EveryVoice.sl/bin/pydantic_cli", line 33, in <module>
    sys.exit(load_entry_point('everyvoice', 'console_scripts', 'pydantic_cli')())
  File "/home/sam037/.conda/envs/EveryVoice.sl/bin/pydantic_cli", line 25, in importlib_load_entry_point
    return next(matches).load()
  File "/home/sam037/.conda/envs/EveryVoice.sl/lib/python3.9/importlib/metadata.py", line 86, in load
    module = import_module(match.group('module'))
  File "/home/sam037/.conda/envs/EveryVoice.sl/lib/python3.9/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1030, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1007, in _find_and_load
  File "<frozen importlib._bootstrap>", line 972, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 228, in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1030, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1007, in _find_and_load
  File "<frozen importlib._bootstrap>", line 986, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 680, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 850, in exec_module
  File "<frozen importlib._bootstrap>", line 228, in _call_with_frames_removed
  File "/fs/hestia_Hnrc/ict/sam037/git/EveryVoice/everyvoice/base_cli/pydantic_cli.py", line 5, in <module>
    from pydantic_cli import run_and_exit, to_runner
  File "/home/sam037/.conda/envs/EveryVoice.sl/lib/python3.9/site-packages/pydantic_cli/__init__.py", line 227, in <module>
    field: pydantic.fields.ModelField,
AttributeError: module 'pydantic.fields' has no attribute 'ModelField'
"""
assert False, "pydantic_cli is not compatible with pydantic>2"

# https://github.com/mpkocher/pydantic-cli#other-related-tools
import sys

from pydantic import BaseModel
from pydantic_cli import run_and_exit, to_runner


class MinOptions(BaseModel):
    input_file: str
    max_records: int


def example_runner(opts: MinOptions) -> int:
    print(f"Mock example running with options {opts}")
    return 0


def cli():
    # to_runner will return a function that takes the args list to run and
    # will return an integer exit code
    sys.exit(to_runner(MinOptions, example_runner, version="0.1.0")(sys.argv[1:]))


if __name__ == "__main__":
    cli()
