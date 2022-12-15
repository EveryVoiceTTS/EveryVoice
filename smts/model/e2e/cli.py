from enum import Enum

import typer
from merge_args import merge_args

from smts.base_cli.interfaces import train_base_command_interface
from smts.model.e2e.config import CONFIGS, SMTSConfig

app = typer.Typer(pretty_exceptions_show_locals=False)

_config_keys = {k: k for k in CONFIGS.keys()}

CONFIGS_ENUM = Enum("CONFIGS", _config_keys)  # type: ignore


@app.command()
@merge_args(train_base_command_interface)
def train(name: CONFIGS_ENUM = typer.Option(None, "--name", "-n"), **kwargs):
    from smts.base_cli.helpers import train_base_command
    from smts.model.e2e.dataset import E2EDataModule
    from smts.model.e2e.model import SmallTeamSpeech

    train_base_command(
        name=name,
        model_config=SMTSConfig,
        configs=CONFIGS,
        model=SmallTeamSpeech,
        data_module=E2EDataModule,
        monitor="validation/mel_spec_error",
        **kwargs
    )
