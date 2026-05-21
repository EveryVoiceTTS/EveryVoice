"""Generic base for synthesis-output-writing Lightning callbacks.

Shared by the FS2 and StyleTTS2 prediction-writing callback hierarchies.
Subclasses override ``on_predict_batch_end`` with format-specific logic.
"""

from __future__ import annotations

from pathlib import Path

from pytorch_lightning.callbacks import Callback


class BasePredictionWritingCallback(Callback):
    """Handles output-directory creation and output-path construction.

    Concrete subclasses must implement ``on_predict_batch_end``.
    """

    def __init__(
        self,
        save_dir: Path,
        file_extension: str,
        global_step: int,
        include_global_step_in_filename: bool = False,
    ) -> None:
        super().__init__()
        self.file_extension = file_extension
        self.global_step = f"ckpt={global_step}"
        self.save_dir = save_dir
        self.sep = "--"
        self.include_global_step_in_filename = include_global_step_in_filename
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def get_filename(self, basename: str, speaker: str, language: str) -> str:
        name_parts = [basename, speaker, language, self.file_extension]
        if self.include_global_step_in_filename:
            name_parts.insert(-1, self.global_step)
        path = self.save_dir / self.sep.join(name_parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
