"""Helpers for resuming training from an experiment's last checkpoint, including
around a Slurm requeue or preemption.

When Slurm sends the requeue signal (SIGUSR1), Lightning saves a checkpoint and
requeues the job. save_requeue_ckpt_to_last() makes Lightning save that
checkpoint as the experiment's last checkpoint, and resolve_resume_ckpt()
decides whether a run resumes from it. check_training_left() checks that
continuing a checkpoint's training state leaves something to train.
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


def check_training_left(
    ckpt: dict,
    ckpt_path: Path | str,
    max_steps: int | None,
    max_epochs: int,
    max_epochs_setting: str = "max_epochs",
) -> None:
    """Exit if the training progress in ckpt already reaches max_steps or
    max_epochs, so that continuing it with training.finetune_mode: full-state
    would train nothing."""
    # ckpt["epoch"] is the epoch the checkpoint was saved in, which is not yet
    # counted as finished when it is saved at the end of that epoch
    epochs_done = (
        ckpt.get("loops", {})
        .get("fit_loop", {})
        .get("epoch_progress", {})
        .get("current", {})
        .get("processed", ckpt.get("epoch", 0))
    )
    steps_done = ckpt.get("global_step", 0)
    steps_reached = max_steps is not None and 0 <= max_steps <= steps_done
    if steps_reached or epochs_done >= max_epochs:
        limits = f"{max_epochs_setting}={max_epochs}"
        if max_steps is not None:
            limits = f"max_steps={max_steps} and {limits}"
        logger.error(
            f"'{ckpt_path}' was saved after {steps_done} steps and {epochs_done} epochs, "
            f"so with {limits} there is nothing left to train. "
            "With training.finetune_mode: full-state, training continues counting from "
            "the checkpoint's progress: raise those limits, or set "
            "training.finetune_mode: weights-only to start counting from 0."
        )
        sys.exit(1)
