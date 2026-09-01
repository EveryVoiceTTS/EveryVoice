"""
Programmatic API to the main EveryVoice commands.

The functions here are wrappers around the similar CLI commands made available for
direct Python scripting, e.g., on HuggingFace.
"""

import multiprocessing as mp
from functools import wraps
from pathlib import Path

from merge_args import merge_args

from everyvoice import cli
from everyvoice.base_cli.helpers import preprocess_base_command
from everyvoice.base_cli.interfaces import preprocess_base_command_interface
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.preprocess import (
    PreprocessCategories,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.preprocess import (
    preprocess as preprocess,
)


# Idea: functools.wrap is precisely intended to merge docs, so this could have worked nicely
# + argument documentation is indeed merged
# - Intellisense shows weird values
# - Intellisense does not actually check if I give this valid parameters
# - no mypy checks
# - no default arg values
@wraps(preprocess_base_command)
def preprocess_wrap_base_cmd(steps: list[str] = [], **kwargs):
    preprocess(steps=[PreprocessCategories(step) for step in steps], **kwargs)


if False:
    preprocess_wrap_base_cmd(
        steps=["audio", "spec"],
        config_file=Path(),
        config_args=[],
        cpus=1,
        overwrite=False,
        debug=False,
        foo="bar",
    )


# Idea: narrow wrap to assign only __doc__
# - Intellisense still shows weird values
# - Intellisense does not actually check if I give this valid parameters
# - no mypy checks
# - no default arg values
# Q: maybe playing with `assigned=` it might be possible to get better results?
@wraps(preprocess_base_command_interface, assigned=["__doc__", "__annotations__"])
def preprocess_wrap_interface(steps=[], **kwargs):
    preprocess(steps=[PreprocessCategories(step) for step in steps], **kwargs)


if False:
    preprocess_wrap_interface(
        steps=["audio"],
        config_file=Path(),
        config_args=[],
        cpus=1,
        overwrite=False,
        debug=False,
        foo="bar",
    )


# Idea: use merge_args like we do to combine base commands
# - Intellisense doesn't work at all
# - no generated documentation
# - no mypy checks
# - no default arg values
@merge_args(preprocess_base_command_interface)
def preprocess_merge_args_interface(**kwargs):
    preprocess(**kwargs)


if False:
    preprocess_merge_args_interface(
        steps=["audio", "spec"],
        config_file=Path(),
        config_args=[],
        cpus=1,
        overwrite=False,
        debug=False,
        foo="bar",
    )


# Idea: use merge_args with the base command instead of the interface
# - Intellisense doesn't work at all
# - no generated documentation
# - no mypy checks
# - no default arg values
@merge_args(preprocess_base_command)
def preprocess_merge_args_base_cmd(**kwargs):
    preprocess(**kwargs)


if False:
    preprocess_merge_args_base_cmd(
        steps=["audio", "spec"],
        config_file=Path(),
        config_args=[],
        cpus=1,
        debug=False,
        foo="bar",
    )


# Rewriting the docs means having the maintain double the documentation, but Intellisense works
# + redeclaring the arguments lets me give them default values
# + Intellisense shows whatever do I write here
# + Intellisense/Pylance checks the arguments correctly
# + mypy checks work
# - not DRY
# - we'll need to make sure the CLI and the API remain in sync
# - no error checking on the arguments beyond just their types
# Mitigations I suggest:
#  - write some tests that make sure this and preprocess remain in sync, at least in terms
#    of the list of parameters they accept, by name, if I can figure it out.
def preprocess_rewrite_docs(
    config_file: Path,
    compute_stats: bool = True,
    steps: list[PreprocessCategories | str] = list(PreprocessCategories),
    config_args: list[str] = [],
    cpus: int = min(4, mp.cpu_count()),
    overwrite: bool = False,
    debug: bool = False,
):
    """
    preprocess

    Args:
      steps: steps to run
    ...
    """
    preprocess(
        compute_stats=compute_stats,
        steps=[PreprocessCategories(step) for step in steps],
        config_file=config_file,
        config_args=config_args,
        cpus=cpus,
        overwrite=overwrite,
        debug=debug,
    )


if False:
    preprocess_rewrite_docs(
        steps=["audio", "foo"], config_file=Path(), config_args=[], foo="bar"
    )


# This approach is inspired from readalongs.api, where instead of trying to call the deep
# function that the CLI eventually calls, we wrap a call to the CLI itself. But with typer
# commands, this does not automate anything for us, neither arg validation nor Intellisense,
# unlike what the similar thing with click does.
# Because we use typer commands, cli.preprocess_fs2 and just fs2.cli.preprocess actually
# behave identically... :(
# + using **kwargs means I don't need to redeclare arguments
# - no intellisense
# - no mypy checking
# - no default values, so I need to add code to set them
def preprocess_cli_wrap(**kwargs):
    preprocess_args = kwargs
    if "config_file" in kwargs:
        preprocess_args["config_file"] = Path(preprocess_args["config_file"])
    else:
        raise ValueError("Missing required config_file name argument")
    preprocess_args.setdefault("config_args", [])
    preprocess_args.setdefault("cpus", min(4, mp.cpu_count()))
    preprocess_args.setdefault("overwrite", False)
    preprocess_args.setdefault("debug", False)

    cli.preprocess_fs2(**preprocess_args)


if False:
    # preprocess_cli_wrap(config_file=".", steps=["audio", "spec"], config_args=[], cpus=1, overwrite=False, debug=False)
    preprocess_cli_wrap(config_file=".", foo="bar")


# Up to this point, all hacks above wrap preprocess() as defined in origin/main, although
# they continue to behave the same with the other changes in my dev.ej/api-hacks branch.

# The next example requires the other changes in dev.ej/api-hacks in both EV and fs2:

# Suggested by AIZone: refactor preprocess itself to stop using merge_args, in which case
# pylance, mypy, Intellisense et all just work out of the box. The Typer CLI is the minimal
# wrapper around the programmatic API instead of the other way around.
# - the definition of preprocess from the base command and interface has more repetition
# + we can reorder inherited parameters in the preprocess definition
# + all static analysis works as expected
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.preprocess import (
    preprocess as preprocess_api,
)

if True:
    preprocess_api(config_file=Path("."), steps=[PreprocessCategories.audio], foo="bar")


# AIZone suggests this core principle:
# - define the core API, then make the CLI a thin shim around it
# And then suggests:
# - instead of merge_args, you can define the parameters using Annotated aliases
#   to reduce how much needs to be retyped
# - alternatively we could define a PreprocessOptions dataclass for the base command parameters,
#   and have preprocess() accept a PreprocessOptions along with the extra options it needs
#   (I'm not sure how this would mesh with Typer for creating the CLI, though)
