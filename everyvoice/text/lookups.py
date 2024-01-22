from typing import Dict, Sequence, Union

from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig


def build_lookup(items: Sequence[Dict[str, str]], key: str) -> Dict[str, int]:
    """
    Create a lookup table from a list of entries and a key into those entries.
    """
    # Using a dictionary instead of a set to preserve the order.
    uniq_items = {item[key]: 1 for item in items}
    return {item: i for i, item in enumerate(uniq_items.keys())}


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
