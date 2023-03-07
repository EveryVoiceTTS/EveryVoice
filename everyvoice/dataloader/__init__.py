import os
from typing import Callable, Optional, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from everyvoice.dataloader.imbalanced_sampler import ImbalancedDatasetSampler
from everyvoice.model.aligner.config import AlignerConfig
from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.model.vocoder.config import VocoderConfig


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: Union[
            AlignerConfig, VocoderConfig, FeaturePredictionConfig, EveryVoiceConfig
        ],
    ):
        super().__init__()
        self.collate_fn: Union[Callable, None] = None
        self.config = config
        self.use_weighted_sampler = False
        self.train_path = os.path.join(
            self.config.training.logger.save_dir,
            self.config.training.logger.name,
            "train_data.pth",
        )
        self.val_path = os.path.join(
            self.config.training.logger.save_dir,
            self.config.training.logger.name,
            "val_data.pth",
        )

    def setup(self, stage: Optional[str] = None):
        # load it back here
        self.train_dataset = torch.load(self.train_path)
        self.val_dataset = torch.load(self.val_path)

    def train_dataloader(self):
        sampler = (
            ImbalancedDatasetSampler(self.train_dataset)
            if self.use_weighted_sampler
            else None
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.config.training.train_data_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
            sampler=sampler,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.train_dataset + self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.config.training.train_data_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        sampler = (
            ImbalancedDatasetSampler(self.val_dataset)
            if self.use_weighted_sampler
            else None
        )
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
            sampler=sampler,
        )

    def prepare_data(self):
        # Override this method
        raise NotImplementedError(
            "This method should be implemented by the child class"
        )

    def load_dataset(self):
        # Override this method
        raise NotImplementedError(
            "The base data module does not have a method implemented for loading a dataset. Please use another Data Loader that inherits the BaseDataModule class."
        )
