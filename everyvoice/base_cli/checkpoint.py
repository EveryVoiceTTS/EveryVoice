"""
CLI command to inspect EveryVoice's checkpoints.
"""

import json
import warnings
from collections import defaultdict
from json import JSONEncoder
from pathlib import Path
from typing import Annotated, Any

import typer

from .interfaces import typer_file_argument

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
            return json.loads(obj.model_dump_json())
        return super().default(obj)


def summarize_statedict(ckpt: dict) -> dict:
    if "state_dict" not in ckpt:
        return {}
    model_keys: dict[str, int] = defaultdict(int)
    for k, v in sorted(ckpt["state_dict"].items()):
        main_key = k.split(".")[0]
        model_keys[main_key] += v.numel()

    model_keys["TOTAL"] = sum(model_keys.values())
    return model_keys


def load_checkpoint(model_path: Path, minimal=True) -> dict[str, Any]:
    """
    Loads a checkpoint and performs minor clean up of the checkpoint.
    Removes the `optimizer_states`'s `state` and `param_groups`'s `params`.
    Removes `state_dict` from the checkpoint.
    """
    import torch

    checkpoint = torch.load(
        model_path, map_location=torch.device("cpu"), weights_only=True
    )

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


def summarize_fs2_model(model_path: Path, checkpoint: dict) -> None:
    from torchinfo import summary

    from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
        FastSpeech2,
    )

    model = FastSpeech2.load_from_checkpoint(model_path)
    print(summary(model, None, verbose=0))


def summarize_hfgl_model(model_path: Path, checkpoint: dict) -> None:
    from torchinfo import summary

    from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import (
        HiFiGAN,
    )

    model = HiFiGAN.load_from_checkpoint(model_path)
    print(summary(model, None, verbose=0))


def summarize_hfgl_generator_model(model_path: Path, checkpoint: dict) -> None:
    import torch
    from torchinfo import summary

    from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
        load_hifigan_from_checkpoint,
    )

    device = torch.device("cpu")
    vocoder_model, vocoder_config = load_hifigan_from_checkpoint(checkpoint, device)

    print(summary(vocoder_model, None, verbose=0))


def summarize_unknown_model(model_path: Path, checkpoint: dict) -> None:
    from tabulate import tabulate

    print(
        "We couldn't read your file, possibly because the version of EveryVoice that created it is incompatible with your installed version."
    )
    if sd_summary := summarize_statedict(checkpoint):
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


@app.command()
def inspect(
    model_path: Annotated[
        Path, typer_file_argument(help="The path to your model checkpoint file.")
    ],
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
        try:
            checkpoint = load_checkpoint(model_path, minimal=True)
        except Exception as e:
            raise ValueError(
                f"Error loading checkpoint '{model_path}'. It might have been created with a different version of EveryVoice that is not compatible."
            ) from e
        config = json.dumps(
            checkpoint,
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
        checkpoint = load_checkpoint(model_path, minimal=False)
        if "model_info" in checkpoint:
            print(
                "Inspecting checkpoint according to its model info:",
                checkpoint["model_info"],
            )
            model_summarizers = {
                "FastSpeech2": summarize_fs2_model,
                "HiFiGAN": summarize_hfgl_model,
                "HiFiGANGenerator": summarize_hfgl_generator_model,
            }
            summarizer = model_summarizers.get(checkpoint["model_info"]["name"], None)
            if summarizer:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        summarizer(model_path, checkpoint)
                    except Exception:
                        summarize_unknown_model(model_path, checkpoint)
                    return

        print("Inspecting unknown model type - trying all known types")
        for summarizer in (
            summarize_hfgl_model,
            summarize_hfgl_generator_model,
            summarize_fs2_model,
            summarize_unknown_model,
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    summarizer(model_path, checkpoint)
                    return
                except Exception:
                    pass


@app.command()
def rename_speaker(
    model_path: Annotated[
        Path, typer_file_argument(help="The path to your model checkpoint file.")
    ],
    old_speaker_name: Annotated[
        str, typer.Argument(help="The name of the speaker to rename.")
    ],
    new_speaker_name: Annotated[
        str, typer.Argument(help="The new name for the speaker.")
    ],
):
    """
    Rename a speaker in the checkpoint's parameters.
    """
    ckpt = load_checkpoint(model_path, minimal=False)

    if (
        "hyper_parameters" in ckpt
        and "speaker2id" in ckpt["hyper_parameters"]
        and len(ckpt["hyper_parameters"]["speaker2id"]) > 0
    ):
        speakers = ckpt["hyper_parameters"]["speaker2id"]
        if old_speaker_name in speakers:
            speakers[new_speaker_name] = speakers.pop(old_speaker_name)
            print(f"Renamed speaker '{old_speaker_name}' to '{new_speaker_name}'.")
            print(f"Updated speakers: {speakers}")
            # Update the parameter in the checkpoint
            ckpt["hyper_parameters"]["speaker2id"] = speakers
            import torch

            torch.save(ckpt, model_path)
            print(f"Updated checkpoint saved to {model_path}.")
        else:
            raise typer.BadParameter(
                f"Speaker '{old_speaker_name}' not found in checkpoint parameters."
            )

    else:
        raise typer.BadParameter("No speakers found in checkpoint parameters.")
