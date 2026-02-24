# flake8: noqa
"""These are function stubs whose sole purpose is to merge with the function signatures of model-specific
cli command functions. Each of these should have a typer default (which can be overidden by the model-specific command)
there should be no body as the union of these signatures and the model-specific signatures is what the helper function
will be called with.
"""
import multiprocessing as mp
from pathlib import Path
from typing import Annotated, Any, Optional

import typer


def typer_file_option(*args, **kwargs) -> Any:
    """Shorthard for setting the typer option parameters to get an existing file."""
    return typer.Option(*args, exists=True, dir_okay=False, file_okay=True, **kwargs)


def typer_file_argument(*args, **kwargs) -> Any:
    """Shorthard for setting the typer argument parameters to get an existing file."""
    return typer.Argument(*args, exists=True, dir_okay=False, file_okay=True, **kwargs)


def load_config_base_command_interface(
    config_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            file_okay=True,
            help="The path to your model configuration file.",
        ),
    ],
    config_args: list[str] = typer.Option(None, "--config", "-c"),
):
    pass


def preprocess_base_command_interface(
    config_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            file_okay=True,
            help="The path to your model configuration file.",
        ),
    ],
    config_args: Annotated[
        list[str],
        typer.Option("-c", "--config-args", help="Override the configuration."),
    ] = [],
    cpus: Annotated[
        Optional[int],
        typer.Option(
            "-C",
            "--cpus",
            help="How many CPUs to use when preprocessing",
        ),
    ] = min(4, mp.cpu_count()),
    overwrite: Annotated[
        bool,
        typer.Option(
            "-O",
            "--overwrite",
            help="Redo all preprocessing, even if files already exist and aren't expected to change.",
        ),
    ] = False,
    debug: Annotated[
        bool, typer.Option("-D", "--debug", help="Enable debugging.")
    ] = False,
):
    pass


def train_base_command_interface(
    config_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            file_okay=True,
            help="The path to your model configuration file.",
        ),
    ],
    config_args: Annotated[
        list[str],
        typer.Option("-c", "--config-args", help="Overwrite the configuration"),
    ] = [],
    accelerator: Annotated[
        str,
        typer.Option(
            "--accelerator",
            "-a",
            help="Uses PyTorch Lightning Accelerators: https://pytorch-lightning.readthedocs.io/en/stable/extensions/accelerator.html",
        ),
    ] = "auto",
    devices: Annotated[
        str, typer.Option("--devices", "-d", help="The number of GPUs on each node")
    ] = "auto",
    nodes: Annotated[
        int, typer.Option("--nodes", "-n", help="The number of nodes on your machine")
    ] = 1,
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            "-s",
            help="The strategy for data parallelization: https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html",
        ),
    ] = "ddp",
):
    pass


def inference_base_command_interface(
    config_args: Annotated[list[str], typer.Option("--config", "-c")] = [],
):
    pass
