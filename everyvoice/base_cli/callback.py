from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from typing_extensions import override


class ResetValidationDataloaderCallback(Callback):
    """
    Reset the validation progress to allow resuming and validating a full
    validation set and not just the first example in the validation set.
    """

    @override
    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> None:
        batch_progress = trainer.fit_loop.epoch_loop.val_loop.batch_progress
        batch_progress.reset()
