"""Generic base for synthesis-output-writing Lightning callbacks.

Used by the StyleTTS2 prediction-writing callbacks, which are driven by
trainer.predict(). Subclasses override ``on_predict_batch_end`` with
format-specific logic. The FS2 writers don't use Lightning callbacks: they
derive from BasePredictionWriter directly.
"""

from pytorch_lightning.callbacks import Callback

from everyvoice.base_cli.prediction_writer import BasePredictionWriter


class BasePredictionWritingCallback(BasePredictionWriter, Callback):
    """BasePredictionWriter as a Lightning Callback.

    Concrete subclasses must implement ``on_predict_batch_end``.
    """
