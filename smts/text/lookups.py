from typing import Union

import pandas as pd

from smts.model.e2e.config import SMTSConfig
from smts.model.feature_prediction.config import FeaturePredictionConfig


class LookupTables:
    def __init__(self, config: Union[SMTSConfig, FeaturePredictionConfig]):
        self.config = config
        self.dataset = self.config.training.filelist_loader(
            self.config.training.filelist
        )
        data_frame = pd.DataFrame(self.dataset)
        speakers = data_frame["speaker"].unique()
        self.speaker2id = {speaker: i for i, speaker in enumerate(speakers)}
        langs = data_frame["language"].unique()
        self.lang2id = {language: i for i, language in enumerate(langs)}
