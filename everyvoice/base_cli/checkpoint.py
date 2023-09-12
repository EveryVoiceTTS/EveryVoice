"""
CLI command to inspect EveryVoice's checkpoints.
"""
import json
import sys
import warnings
from enum import Enum
from json import JSONEncoder
from pathlib import Path
from typing import Any, Dict

import typer
import yaml
from pydantic import BaseModel
from typing_extensions import Annotated

from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
    FastSpeech2,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import HiFiGAN

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="Extract checkpoint's hyperparameters.",
)


class ExportType(str, Enum):
    """
    Available export format for the configuration.
    """

    JSON = "json"
    YAML = "yaml"


class CheckpointEncoder(JSONEncoder):
    """
    Helper JSON Encoder for missing `torch.Tensor` & `pydantic.BaseModel`.
    """

    def default(self, obj: Any):
        """
        Extends json to handle `torch.Tensor` and `pydantic.BaseModel`.
        """
        import torch

        if isinstance(obj, torch.Tensor):
            return list(obj.shape)
        elif isinstance(obj, BaseModel):
            return json.loads(obj.json())
        return super().default(obj)


def load_checkpoint(model_path: Path) -> Dict[str, Any]:
    """
    Loads a checkpoint and performs minor clean up of the checkpoint.
    Removes the `optimizer_states`'s `state` and `param_groups`'s `params`.
    Removes `state_dict` from the checkpoint.
    """
    import torch

    checkpoint = torch.load(str(model_path), map_location=torch.device("cpu"))

    # Some clean up of useless stuff.
    if "optimizer_states" in checkpoint:
        for optimizer in checkpoint["optimizer_states"]:
            # Delete the optimizer history values.
            if "state" in optimizer:
                del optimizer["state"]
            # These are simply values [0, len(checkpoint["optimizer_states"][0]["state"])].
            for param_group in optimizer["param_groups"]:
                if "params" in param_group:
                    del param_group["params"]

    if "state_dict" in checkpoint:
        del checkpoint["state_dict"]

    if "loops" in checkpoint:
        del checkpoint["loops"]

    return checkpoint


@app.command()
def inspect(
    model_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to your model checkpoint file.",
    ),
    export_type: ExportType = ExportType.YAML,
    show_config: Annotated[
        bool,
        typer.Option(
            "--show-config/--no-show-config",  # noqa
            "-c/-C",  # noqa
            help="Show the configuration used during training in either json or yaml format",  # noqa
        ),
    ] = True,
    show_architecture: Annotated[
        bool,
        typer.Option(
            "--show-architecture/--no-show-architecture",  # noqa
            "-a/-A",  # noqa
            help="Show the model's architecture",  # noqa
        ),
    ] = True,
    show_weights: Annotated[
        bool,
        typer.Option(
            "--show-weights/--no-show-weights",  # noqa
            "-w/-W",  # noqa
            help="Show the number of weights per layer",  # noqa
        ),
    ] = True,
):
    """
    Given an EveryVoice checkpoint, show information about the configuration
    used during training, the model's architecture and the number of weights
    per layer and total weight count.
    """
    checkpoint = load_checkpoint(model_path)

    if show_config:
        print("Configs:")
        if export_type is ExportType.JSON:
            json.dump(
                checkpoint,
                sys.stdout,
                ensure_ascii=False,
                indent=2,
                cls=CheckpointEncoder,
            )
        elif export_type is ExportType.YAML:
            output = json.loads(json.dumps(checkpoint, cls=CheckpointEncoder))
            yaml.dump(output, stream=sys.stdout)
        else:
            raise NotImplementedError(f"Unsupported export type {export_type}!")

    if show_architecture:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = HiFiGAN.load_from_checkpoint(model_path)
            # NOTE if ANY exception is raise, that means the model couldn't be
            # loaded and we want to try another config type.  This is to "ask
            # forgiveness, not permission".
            except Exception:
                try:
                    model = FastSpeech2.load_from_checkpoint(model_path)
                except Exception:
                    raise NotImplementedError(
                        "Your checkpoint contains a model type that is not yet supported!"
                    )
            print("\n\nModel Architecture:\n", model, sep="")

    if show_weights:
        from torchinfo import summary

        statistics = summary(model, None, verbose=0)
        print("\nModel's Weights:\n", statistics)
        # According to Aidan (1, 80, 50) should be a valid input size but it looks
        # like the model is expecting a Dict which isn't supported by torchsummary.
        # print(summary(model, (1, 80, 50)))
