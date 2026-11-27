"""Atomic checkpoint writes, so an interrupted save never truncates the existing checkpoint."""

import os
from pathlib import Path
from typing import Any


def save_atomically(checkpoint: dict[str, Any], path: Path | str) -> None:
    """Save checkpoint to path, replacing any existing file only once the new one is complete.

    The checkpoint is written to a temporary file in the same directory, synced to
    disk, then renamed over path. A rename within one filesystem is atomic, unlike
    a move from the system temp dir, which can be a copy.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            torch.save(checkpoint, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_checkpoint_io(torch_checkpoint_io_cls):
    """Return a CheckpointIO plugin that saves checkpoints with save_atomically().

    torch_checkpoint_io_cls is the TorchCheckpointIO of the Lightning namespace
    whose Trainer receives the plugin (pytorch_lightning or lightning.pytorch):
    a Trainer only accepts plugins from its own namespace.
    """

    class AtomicCheckpointIO(torch_checkpoint_io_cls):
        def save_checkpoint(self, checkpoint, path, storage_options=None):
            if storage_options is not None:
                raise TypeError(
                    f"`storage_options` is not supported by `{self.__class__.__name__}`."
                )
            save_atomically(checkpoint, path)

    return AtomicCheckpointIO()
