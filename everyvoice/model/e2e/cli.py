import typer
from merge_args import merge_args

from everyvoice.base_cli.interfaces import train_base_command_interface

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="End-to-end training: jointly train the FastSpeech2 and HiFiGAN networks",
)


@app.command()
@merge_args(train_base_command_interface)
def train(**kwargs):
    from everyvoice.base_cli.helpers import train_base_command
    from everyvoice.model.e2e.config import EveryVoiceConfig
    from everyvoice.model.e2e.dataset import E2EDataModule
    from everyvoice.model.e2e.model import EveryVoice

    train_base_command(
        model_config=EveryVoiceConfig,
        model=EveryVoice,
        data_module=E2EDataModule,
        monitor="validation/mel_spec_error",
        gradient_clip_val=None,
        **kwargs,
    )
