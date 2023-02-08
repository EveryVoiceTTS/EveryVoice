import os
import random
from pathlib import Path
from typing import Callable, Optional, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from smts.dataloader.imbalanced_sampler import ImbalancedDatasetSampler
from smts.model.aligner.config import AlignerConfig
from smts.model.e2e.config import SMTSConfig
from smts.model.feature_prediction.config import FeaturePredictionConfig
from smts.model.vocoder.config import VocoderConfig


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: Union[
            AlignerConfig, VocoderConfig, FeaturePredictionConfig, SMTSConfig
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
            shuffle=False
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


class AudioDataset(Dataset):
    def __init__(
        self, audio_files, config: VocoderConfig, use_segments=False, segment_size=None
    ):
        self.config = config
        self.sep = config.preprocessing.value_separator
        self.use_segments = use_segments
        self.audio_files = audio_files
        self.preprocessed_dir = Path(self.config.preprocessing.save_dir)
        self.segment_size = segment_size
        self.sampling_rate = self.config.preprocessing.audio.input_sampling_rate
        
    def __getitem__(self, index):
        """
        y = waveform from potentially upsampled audio
        """
        item = self.audio_files[index]
        speaker = "default" if "speaker" not in item else item["speaker"]
        language = "default" if "language" not in item else item["language"]
        y = torch.load(
            self.preprocessed_dir
            / "audio"
            / self.sep.join(
                [
                    item["basename"],
                    speaker,
                    language,
                    f"audio-{self.sampling_rate}.pt",
                ]
            )
        ).squeeze()  # [samples] should be output sample rate, squeeze to get rid of channels just in case
  
        if self.use_segments:
            if y.size(0) >= self.segment_size:
                max_start = random.randint(0, len(y) - self.segment_size - 1)
                y = y[
                    max_start : max_start + self.segment_size,
                ]
            else:
                y = torch.nn.functional.pad(y, (0, self.segment_size - y.size(0)), "constant")

        return (y, self.audio_files[index]["basename"])

    def __len__(self):
        return len(self.audio_files)

    def get_labels(self):
        return [x["label"] for x in self.audio_files]


class OODDataModule(BaseDataModule):
    def __init__(self, config: VocoderConfig, use_segments=False, segment_size=None):
        super().__init__(config=config)
        self.use_weighted_sampler = config.training.use_weighted_sampler
        self.batch_size = config.training.batch_size
        self.train_split = self.config.training.train_split
        self.use_segments = use_segments
        self.segment_size = segment_size
        self.train_path = os.path.join(
            self.config.training.logger.save_dir,
            self.config.training.logger.name,
            "ood_train_data.pth",
        )
        self.val_path = os.path.join(
            self.config.training.logger.save_dir,
            self.config.training.logger.name,
            "ood_val_data.pth",
        )
        if self.use_segments and self.segment_size is None:
            raise ValueError("You must set a segment size if you use segments")
        if not self.use_segments and self.segment_size is not None:
            raise ValueError(f"You must set the use_segments flag if you want to segment your audio to size {self.segment_size}")
        if self.segment_size is not None:
            self.config.preprocessing.audio.vocoder_segment_size = self.segment_size

    def load_dataset(self):
        self.dataset = self.config.training.filelist_loader(
            self.config.training.filelist
        )

    def prepare_data(self):
        self.load_dataset()
        self.dataset_length = len(self.dataset)
        train_samples = int(self.dataset_length * self.train_split)
        val_samples = self.dataset_length - train_samples
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_samples, val_samples]
        )
        self.train_dataset = AudioDataset(
            self.train_dataset, self.config, use_segments=self.use_segments, segment_size=self.segment_size
        )
        self.val_dataset = AudioDataset(
            self.val_dataset, self.config, use_segments=self.use_segments, segment_size=self.segment_size
        )
        # save it to disk
        torch.save(self.train_dataset, self.train_path)
        torch.save(self.val_dataset, self.val_path)