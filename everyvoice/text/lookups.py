from typing import Union

import pandas as pd

from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig


class LookupTables:
    def __init__(self, config: Union[EveryVoiceConfig, FeaturePredictionConfig]):
        self.config = config
        self.val_dataset = self.config.training.filelist_loader(
            self.config.training.validation_filelist
        )
        self.train_dataset = self.config.training.filelist_loader(
            self.config.training.training_filelist
        )
        data_frame = pd.DataFrame(self.train_dataset + self.val_dataset)
        speakers = data_frame["speaker"].unique()
        self.speaker2id = {speaker: i for i, speaker in enumerate(speakers)}
        langs = data_frame["language"].unique()
        self.lang2id = {language: i for i, language in enumerate(langs)}
