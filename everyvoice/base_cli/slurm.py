"""Helpers for resuming training from an experiment's last checkpoint, including
around a Slurm requeue or preemption.

When Slurm sends the requeue signal (SIGUSR1), Lightning saves a checkpoint and
requeues the job. save_requeue_ckpt_to_last() makes Lightning save that
checkpoint as the experiment's last checkpoint, and resolve_resume_ckpt()
decides whether a run resumes from it.
"""

import os
import sys
from pathlib import Path

from everyvoice import logger


def slurm_restarted() -> bool:
    """True when Slurm has requeued this job at least once."""
    return int(os.environ.get("SLURM_RESTART_COUNT", 0) or 0) > 0


def last_ckpt_path(last_ckpt_callback) -> str:
    """The path a ModelCheckpoint saves its "last" checkpoint to.

    The callback's dirpath is resolved during the setup of trainer.fit().
    """
    cb = last_ckpt_callback
    return os.path.join(cb.dirpath, cb.CHECKPOINT_NAME_LAST + cb.FILE_EXTENSION)


def save_requeue_ckpt_to_last(trainer, last_ckpt_callback) -> None:
    """Make Lightning save its Slurm requeue checkpoint as last_ckpt_callback's
    last checkpoint, so the requeued job resumes from it with --auto-resume.

    Lightning's SIGUSR1 handler calls trainer._checkpoint_connector.hpc_save_path()
    to get the path it saves to, then requeues the job.
    """

    def hpc_save_path(_folder) -> str:
        path = last_ckpt_path(last_ckpt_callback)
        if trainer.is_global_zero:
            logger.info(
                f"Slurm requeue signal received: saving checkpoint at step {trainer.global_step} to {path}"
            )
        return path

    trainer._checkpoint_connector.hpc_save_path = hpc_save_path


def resolve_resume_ckpt(
    auto_resume: bool,
    last_ckpt: Path | str,
    finetune_checkpoint: Path | str | None = None,
) -> Path | None:
    """Return the checkpoint to resume this experiment from, or None to start a new run.

    Resuming is on when auto_resume is set or Slurm has requeued the job.
      - Resuming and last_ckpt exists: resume from last_ckpt, even if
        finetune_checkpoint is set.
      - Resuming and last_ckpt doesn't exist: start a new run.
      - Not resuming and last_ckpt exists: exit, since the run would overwrite it.
      - Not resuming and last_ckpt doesn't exist: start a new run.
    """
    last_ckpt = Path(last_ckpt)
    if not auto_resume and slurm_restarted():
        logger.info(
            "Slurm requeued this job (SLURM_RESTART_COUNT > 0): resuming as if --auto-resume was passed."
        )
        auto_resume = True

    if auto_resume:
        if last_ckpt.exists():
            logger.info(f"Resuming from checkpoint '{last_ckpt}'")
            return last_ckpt
        logger.info(f"No checkpoint at '{last_ckpt.resolve()}'; starting a new run.")
        return None

    if last_ckpt.exists():
        if finetune_checkpoint is not None and (
            Path(finetune_checkpoint).resolve() == last_ckpt.resolve()
        ):
            logger.error(
                f"finetune_checkpoint '{finetune_checkpoint}' is this experiment's own last checkpoint, "
                "and training would overwrite it. Fine-tuning needs a new experiment folder: "
                "set a different training.logger.version (e.g. -c training.logger.version=finetuned)."
            )
        else:
            logger.error(
                f"This experiment already has a checkpoint at '{last_ckpt}', and training would overwrite it. "
                "To continue that run, pass --auto-resume. To start a new run, "
                "set a different training.logger.version or move the old checkpoint away."
            )
        sys.exit(1)
    return None
