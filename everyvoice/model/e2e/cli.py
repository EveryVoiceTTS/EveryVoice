from enum import Enum

import typer
from merge_args import merge_args

from everyvoice.base_cli.interfaces import train_base_command_interface
from everyvoice.model.e2e.config import CONFIGS, EveryVoiceConfig

app = typer.Typer(pretty_exceptions_show_locals=False)

_config_keys = {k: k for k in CONFIGS.keys()}

CONFIGS_ENUM = Enum("CONFIGS", _config_keys)  # type: ignore


@app.command()
@merge_args(train_base_command_interface)
def train(name: CONFIGS_ENUM = typer.Option(None, "--name", "-n"), **kwargs):
    from everyvoice.base_cli.helpers import train_base_command
    from everyvoice.model.e2e.dataset import E2EDataModule
    from everyvoice.model.e2e.model import EveryVoice

    train_base_command(
        name=name,
        model_config=EveryVoiceConfig,
        configs=CONFIGS,
        model=EveryVoice,
        data_module=E2EDataModule,
        monitor="validation/mel_spec_error",
        **kwargs
    )
