"""Helpers for saving and resuming checkpoints around a Slurm requeue/preemption.

Lightning saves an hpc_ckpt_N.ckpt when Slurm sends the requeue signal (SIGUSR1),
into trainer.default_root_dir. These helpers put those checkpoints in a per-job
folder under the checkpoints directory and include the global step in their names.
"""

import os
import re
from pathlib import Path

from everyvoice import logger

HPC_CKPT_RE = re.compile(r"^hpc_ckpt_(\d+)(-step=\d+)?\.ckpt$")


def slurm_job_ckpt_dir(ckpt_dir: Path | str) -> Path | None:
    """Return <ckpt_dir>/<SLURM_JOB_ID> when running under Slurm, else None.

    SLURM_JOB_ID is unchanged by `scontrol requeue`, so a requeued job finds the
    checkpoints it saved before being requeued.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    return Path(ckpt_dir) / job_id if job_id else None


def _hpc_ckpts(folder: Path | str) -> dict[int, Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return {}
    return {
        int(match.group(1)): path
        for path in folder.iterdir()
        if (match := HPC_CKPT_RE.match(path.name))
    }


def name_hpc_ckpts_by_step(trainer) -> None:
    """Name the checkpoints Lightning saves on the Slurm requeue signal
    hpc_ckpt_N-step=S.ckpt instead of hpc_ckpt_N.ckpt, and log each save."""

    def hpc_save_path(folder) -> str:
        n = max(_hpc_ckpts(folder), default=0) + 1
        path = Path(folder) / f"hpc_ckpt_{n}-step={trainer.global_step}.ckpt"
        if trainer.is_global_zero:
            logger.info(
                f"Slurm requeue signal received: saving checkpoint at step {trainer.global_step} to {path}"
            )
        return str(path)

    # Lightning's SIGUSR1 handler calls this static method on the trainer's instance
    trainer._checkpoint_connector.hpc_save_path = hpc_save_path


def latest_hpc_ckpt(folder: Path | str | None) -> Path | None:
    """Return the most recent hpc_ckpt_N[-step=S].ckpt in folder, if any."""
    if folder is None:
        return None
    ckpts = _hpc_ckpts(folder)
    return ckpts[max(ckpts)] if ckpts else None
