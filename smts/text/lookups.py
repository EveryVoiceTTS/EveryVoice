import pandas as pd

from smts.config.base_config import BaseConfig


class LookupTables:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.dataset = self.config["training"]["feature_prediction"]["filelist_loader"](
            self.config["training"]["feature_prediction"]["filelist"]
        )
        data_frame = pd.DataFrame(self.dataset)
        speakers = data_frame["speaker"].unique()
        self.speaker2id = {speaker: i for i, speaker in enumerate(speakers)}
        langs = data_frame["language"].unique()
        self.lang2id = {language: i for i, language in enumerate(langs)}
