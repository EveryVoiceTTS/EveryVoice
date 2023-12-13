from typing import Dict, Sequence, Union

from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig


def build_lookup(items: Sequence[Dict[str, str]], key: str) -> Dict[str, int]:
    """
    Create a lookup table from a list of entries and a key into those entries.
    """
    uniq_items = set((item[key] for item in items))
    return {item: i for i, item in enumerate(sorted(uniq_items))}


class LookupTables:
    def __init__(self, config: Union[EveryVoiceConfig, FeaturePredictionConfig]):
        self.config = config
        self.val_dataset = self.config.training.filelist_loader(
            self.config.training.validation_filelist
        )
        self.train_dataset = self.config.training.filelist_loader(
            self.config.training.training_filelist
        )
        self.speaker2id = build_lookup(self.train_dataset + self.val_dataset, "speaker")
        self.lang2id = build_lookup(self.train_dataset + self.val_dataset, "language")
