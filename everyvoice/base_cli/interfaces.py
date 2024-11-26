# flake8: noqa
"""These are function stubs whose sole purpose is to merge with the function signatures of model-specific
    cli command functions. Each of these should have a typer default (which can be overidden by the model-specific command)
    there should be no body as the union of these signatures and the model-specific signatures is what the helper function
    will be called with.
"""
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional

import typer


def complete_path(ctx, param, incomplete) -> list[str]:
    # https://github.com/tiangolo/typer/discussions/625
    # Work-around for path completion bug in CLI shell_complete
    return []


def load_config_base_command_interface(
    config_file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to your model configuration file.",
        shell_complete=complete_path,
    ),
    config_args: List[str] = typer.Option(None, "--config", "-c"),
):
    pass


def preprocess_base_command_interface(
    config_file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to your model configuration file.",
        shell_complete=complete_path,
    ),
    config_args: List[str] = typer.Option(
        None, "-c", "--config-args", help="Override the configuration."
    ),
    cpus: Optional[int] = typer.Option(
        min(4, mp.cpu_count()),
        "-C",
        "--cpus",
        help="How many CPUs to use when preprocessing",
    ),
    overwrite: bool = typer.Option(
        False,
        "-O",
        "--overwrite",
        help="Redo all preprocessing, even if files already exist and aren't expected to change.",
    ),
    debug: bool = typer.Option(False, "-D", "--debug", help="Enable debugging."),
):
    pass


def train_base_command_interface(
    config_file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to your model configuration file.",
        shell_complete=complete_path,
    ),
    config_args: List[str] = typer.Option(
        None, "-c", "--config-args", help="Overwrite the configuration"
    ),
    accelerator: str = typer.Option(
        "auto",
        "--accelerator",
        "-a",
        help="Uses PyTorch Lightning Accelerators: https://pytorch-lightning.readthedocs.io/en/stable/extensions/accelerator.html",
    ),
    devices: str = typer.Option(
        "auto", "--devices", "-d", help="The number of GPUs on each node"
    ),
    nodes: int = typer.Option(
        1, "--nodes", "-n", help="The number of nodes on your machine"
    ),
    strategy: str = typer.Option(
        "ddp",
        "--strategy",
        "-s",
        help="The strategy for data parallelization: https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html",
    ),
):
    pass


def inference_base_command_interface(
    config_file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to your model configuration file.",
        shell_complete=complete_path,
    ),
    config_args: List[str] = typer.Option(None, "--config", "-c"),
    accelerator: str = typer.Option(
        "auto",
        "--accelerator",
        "-a",
        help="Uses PyTorch Lightning Accelerators: https://pytorch-lightning.readthedocs.io/en/stable/extensions/accelerator.html",
    ),
    devices: str = typer.Option(
        "auto", "--devices", "-d", help="The number of GPUs on each node"
    ),
    nodes: int = typer.Option(
        1, "--nodes", "-n", help="The number of nodes on your machine"
    ),
    strategy: str = typer.Option(
        "ddp",
        "--strategy",
        "-s",
        help="The strategy for data parallelization: https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html",
    ),
):
    pass
