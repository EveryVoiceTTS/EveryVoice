import math
import os
import random
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from config.base_config import BaseConfig


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self.train_path = os.path.join(
            self.config["training"]["logger"]["save_dir"], "train_data.pth"
        )
        self.val_path = os.path.join(
            self.config["training"]["logger"]["save_dir"], "val_data.pth"
        )

    def prepare_data(self):
        dataset = self.load_dataset()
        train_split = int(len(dataset) * self.config["training"]["train_split"])
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_split, len(dataset) - train_split]
        )
        # save it to disk
        torch.save(self.train_dataset, self.train_path)
        torch.save(self.val_dataset, self.val_path)

    def setup(self, stage: Optional[str] = None):
        # load it back here
        self.train_dataset = torch.load(self.train_path)
        self.val_dataset = torch.load(self.val_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.config["training"]["batch_size"]
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.config["training"]["batch_size"]
        )

    def load_dataset(self):
        # Override this method
        pass


class SpecDataset(Dataset):
    def __init__(self, config, finetune=False):
        self.config = config
        self.audio_files = self.config["training"]["vocoder"]["filelist_loader"](
            self.config["training"]["vocoder"]["filelist"]
        )
        self.preprocessed_dir = self.config["preprocessing"]["save_dir"]
        self.finetune = finetune
        random.seed(self.config["training"]["vocoder"]["seed"])
        self.segment_size = self.config["preprocessing"]["audio"][
            "vocoder_segment_size"
        ]
        self.hop_size = self.config["preprocessing"]["audio"]["fft_hop_frames"]

    def __getitem__(self, index):
        audio = torch.load(
            os.path.join(self.preprocessed_dir, f"{self.audio_files[index]}-audio.npy")
        )
        # TODO: add speaker/language
        spec = spec_from_audio = torch.load(
            os.path.join(
                self.preprocessed_dir,
                f"{self.audio_files[index]}-spec-{self.config['preprocessing']['audio']['spec_type']}.npy",
            )
        )

        if self.finetune:
            # If finetuning, use the synthesized spectral features
            # TODO: add speaker/language
            spec = torch.load(
                os.path.join(
                    self.preprocessed_dir,
                    f"{self.audio_files[index]}-spec-synthesized.npy",
                )
            )

        frames_per_seg = math.ceil(self.segment_size / self.hop_size)
        if audio.size(1) >= self.segment_size:
            max_spec_start = spec.size(2) - frames_per_seg - 1
            spec_start = random.randint(0, max_spec_start)
            spec = spec[:, :, spec_start : spec_start + frames_per_seg]
            spec_from_audio = spec_from_audio[
                :, :, spec_start : spec_start + frames_per_seg
            ]
            audio = audio[
                :,
                spec_start
                * self.hop_size : (spec_start + frames_per_seg)
                * self.hop_size,
            ]
        else:
            spec = torch.nn.functional.pad(
                spec, (0, frames_per_seg - spec.size(2)), "constant"
            )
            spec_from_audio = torch.nn.functional.pad(
                spec_from_audio,
                (0, frames_per_seg - spec_from_audio.size(2)),
                "constant",
            )
            audio = torch.nn.functional.pad(
                audio, (0, self.segment_size - audio.size(1)), "constant"
            )

        return (spec, audio, self.audio_files[index], spec_from_audio)

    def __len__(self):
        return len(self.audio_files)


class HiFiGANDataModule(BaseDataModule):
    def load_dataset(self):
        self.dataset = SpecDataset(config=self.config)


class HiFiGANFineTuneDataModule(BaseDataModule):
    def load_dataset(self):
        self.dataset = SpecDataset(config=self.config, finetune=True)


class FeaturePredictionDataModule(BaseDataModule):
    def load_dataset(self):
        pass


class E2EDataModule(BaseDataModule):
    def load_dataset(self):
        pass
