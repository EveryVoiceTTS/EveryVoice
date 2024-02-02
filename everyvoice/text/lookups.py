from itertools import chain
from typing import Dict, Iterable, Sequence, Tuple, Union

from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig

LookupTable = dict[str, int]


def lookuptables_from_config(
    config: Union[EveryVoiceConfig, FeaturePredictionConfig]
) -> Tuple[LookupTable, LookupTable]:
    """ """
    train_dataset = config.training.filelist_loader(config.training.training_filelist)
    val_dataset = config.training.filelist_loader(config.training.validation_filelist)

    return lookuptables_from_data((train_dataset, val_dataset))


def lookuptables_from_data(
    data: Iterable[Sequence[Dict[str, str]]]
) -> Tuple[LookupTable, LookupTable]:
    """ """
    languages = set(d["language"] for d in chain(*data) if "language" in d)
    lang2id = {language: i for i, language in enumerate(sorted(languages))}

    speakers = set(d["speaker"] for d in chain(*data) if "speaker" in d)
    speaker2id = {speaker: i for i, speaker in enumerate(sorted(speakers))}

    return lang2id, speaker2id


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
        self.lang2id, self.speaker2id = lookuptables_from_config(self.config)
