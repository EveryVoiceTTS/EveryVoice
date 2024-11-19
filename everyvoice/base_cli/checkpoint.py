"""
CLI command to inspect EveryVoice's checkpoints.
"""

import json
import warnings
from collections import defaultdict
from json import JSONEncoder
from pathlib import Path
from typing import Any, Dict

import typer
from typing_extensions import Annotated

from everyvoice.base_cli.interfaces import complete_path

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="Extract checkpoint's hyperparameters.",
)


class CheckpointEncoder(JSONEncoder):
    """
    Helper JSON Encoder for missing `torch.Tensor` & `pydantic.BaseModel`.
    """

    def default(self, obj: Any):
        """
        Extends json to handle `torch.Tensor` and `pydantic.BaseModel`.
        """
        import torch
        from pydantic import BaseModel

        if isinstance(obj, torch.Tensor):
            return list(obj.shape)
        elif isinstance(obj, BaseModel):
            return json.loads(obj.json())
        return super().default(obj)


def summarize_statedict(ckpt: dict) -> dict:
    if "state_dict" not in ckpt:
        return {}
    model_keys: Dict[str, int] = defaultdict(int)
    for k, v in sorted(ckpt["state_dict"].items()):
        main_key = k.split(".")[0]
        model_keys[main_key] += v.numel()

    model_keys["TOTAL"] = sum(model_keys.values())
    return model_keys


def load_checkpoint(model_path: Path, minimal=True) -> Dict[str, Any]:
    """
    Loads a checkpoint and performs minor clean up of the checkpoint.
    Removes the `optimizer_states`'s `state` and `param_groups`'s `params`.
    Removes `state_dict` from the checkpoint.
    """
    import torch

    checkpoint = torch.load(str(model_path), map_location=torch.device("cpu"))

    if minimal:
        # Some clean up of useless stuff.
        if "optimizer_states" in checkpoint:
            for optimizer in checkpoint["optimizer_states"]:
                # Delete the optimizer history values.
                if "state" in optimizer:
                    del optimizer["state"]
                # These are simply values [0, len(checkpoint["optimizer_states"][0]["state"])].
                if "param_groups" in optimizer:
                    for param_group in optimizer["param_groups"]:
                        if "params" in param_group:
                            del param_group["params"]

        if "state_dict" in checkpoint:
            del checkpoint["state_dict"]

        if "callbacks" in checkpoint:
            del checkpoint["callbacks"]

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
        shell_complete=complete_path,
    ),
    show_config: Annotated[
        bool,
        typer.Option(
            "--show-config/--no-show-config",  # noqa
            "-c/-C",  # noqa
            help="Show the configuration used during training",  # noqa
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
):
    """
    Given an EveryVoice checkpoint, show information about the configuration
    used during training, the model's architecture and the number of weights
    per layer and total weight count.
    """

    if show_config:
        config = json.dumps(
            load_checkpoint(model_path),
            ensure_ascii=False,
            indent=2,
            cls=CheckpointEncoder,
        )
        print(
            """
                ++++++++++++++
                    CONFIG
                ++++++++++++++
            """
        )
        print(config)

    if show_architecture:
        from torchinfo import summary

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model: Any  # HiFiGAN | FastSpeech2 | dict[str, Any] but no import for speed
            try:
                from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import (
                    HiFiGAN,
                )

                model = HiFiGAN.load_from_checkpoint(model_path)
                print(summary(model, None, verbose=0))
            # NOTE if ANY exception is raise, that means the model couldn't be
            # loaded and we want to try another config type.  This is to "ask
            # forgiveness, not permission".
            except Exception:
                try:
                    from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
                        FastSpeech2,
                    )

                    model = FastSpeech2.load_from_checkpoint(model_path)
                    print(summary(model, None, verbose=0))
                except Exception:
                    from tabulate import tabulate

                    model = load_checkpoint(model_path, minimal=False)
                    print(
                        "We couldn't read your file, possibly because the version of EveryVoice that created it is incompatible with your installed version."
                    )
                    if sd_summary := summarize_statedict(model):
                        print(
                            f"We've tried to infer some information from your checkpoint. It appears to have {round(sd_summary['TOTAL'] / 1000000, 2)} M parameters."
                        )
                        print(
                            tabulate(
                                [[k, v] for k, v in sd_summary.items()],
                                tablefmt="rounded_grid",
                                intfmt=",",
                                headers=["LayerName", "Number of Parameters"],
                            )
                        )
