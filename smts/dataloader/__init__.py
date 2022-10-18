import math
import os
import random
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from smts.config.base_config import BaseConfig
from smts.dataloader.imbalanced_sampler import ImbalancedDatasetSampler


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self.use_weighted_sampler = False
        self.train_path = os.path.join(
            self.config["training"]["logger"]["save_dir"],
            self.config["training"]["logger"]["name"],
            "train_data.pth",
        )
        self.val_path = os.path.join(
            self.config["training"]["logger"]["save_dir"],
            self.config["training"]["logger"]["name"],
            "val_data.pth",
        )

    def prepare_data(self):
        self.load_dataset()
        train_split = int(len(self.dataset) * self.config["training"]["train_split"])
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_split, len(self.dataset) - train_split]
        )
        self.train_dataset = SpecDataset(
            self.train_dataset, self.config, use_segments=True
        )
        self.val_dataset = SpecDataset(
            self.val_dataset, self.config, use_segments=False
        )
        # save it to disk
        torch.save(self.train_dataset, self.train_path)
        torch.save(self.val_dataset, self.val_path)

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
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["training"]["train_data_workers"],
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
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
            num_workers=1,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
        )

    def load_dataset(self):
        # Override this method
        raise NotImplementedError(
            "The base data module does not have a method implemented for loading a dataset. Please use another Data Loader that inherits the BaseDataModule class."
        )


class SpecDataset(Dataset):
    def __init__(self, audio_files, config, use_segments=False, finetune=False):
        self.config = config
        self.sep = config["preprocessing"]["value_separator"]
        self.use_segments = use_segments
        self.audio_files = audio_files
        self.preprocessed_dir = Path(self.config["preprocessing"]["save_dir"])
        self.finetune = finetune
        random.seed(self.config["training"]["vocoder"]["seed"])
        self.segment_size = self.config["preprocessing"]["audio"][
            "vocoder_segment_size"
        ]
        self.output_sampling_rate = self.config["preprocessing"]["audio"][
            "output_sampling_rate"
        ]
        self.input_sampling_rate = self.config["preprocessing"]["audio"][
            "input_sampling_rate"
        ]
        self.sampling_rate_change = (
            self.output_sampling_rate // self.input_sampling_rate
        )
        self.input_hop_size = self.config["preprocessing"]["audio"]["fft_hop_frames"]
        self.output_hop_size = (
            self.config["preprocessing"]["audio"]["fft_hop_frames"]
            * self.sampling_rate_change
        )

    def __getitem__(self, index):
        """
        x = mel spectrogram from potentially downsampled audio or from acoustic feature prediction
        y = waveform from potentially upsampled audio
        y_mel = mel spectrogram calculated from y
        """
        item = self.audio_files[index]
        speaker = "default" if "speaker" not in item else item["speaker"]
        language = "default" if "language" not in item else item["language"]
        y = torch.load(
            self.preprocessed_dir
            / self.sep.join(
                [
                    item["basename"],
                    speaker,
                    language,
                    f"audio-{self.output_sampling_rate}.npy",
                ]
            )
        )  # [samples] should be output sample rate
        y_mel = torch.load(
            self.preprocessed_dir
            / self.sep.join(
                [
                    item["basename"],
                    speaker,
                    language,
                    f"spec-{self.output_sampling_rate}-{self.config['preprocessing']['audio']['spec_type']}.npy",
                ]
            )
        )  # [mel_bins, frames]
        if self.finetune:
            # If finetuning, use the synthesized spectral features
            x = torch.load(
                self.preprocessed_dir
                / self.sep.join(
                    [item["basename"], speaker, language, "spec-synthesized.npy"]
                )
            )
        else:
            x = torch.load(
                self.preprocessed_dir
                / self.sep.join(
                    [
                        item["basename"],
                        speaker,
                        language,
                        f"spec-{self.input_sampling_rate}-{self.config['preprocessing']['audio']['spec_type']}.npy",
                    ]
                )
            )  # [mel_bins, frames]
        frames_per_seg = math.ceil(
            self.segment_size / self.output_hop_size
        )  # segment size is relative to output_sampling_rate, so we use the output_hop_size, but frames_per_seg is in frequency domain, so invariant to x and y_mel
        # other implementations just resample y and take the mel spectrogram of that, but this solution allows for segmenting predicted mel spectrograms from the acoustic feature prediction network too

        if self.use_segments:
            # randomly select a segment, if the segment is too short, pad it with zeros
            if y.size(0) >= self.segment_size:
                max_spec_start = x.size(1) - frames_per_seg - 1
                spec_start = random.randint(0, max_spec_start)
                x = x[:, spec_start : spec_start + frames_per_seg]
                y_mel = y_mel[:, spec_start : spec_start + frames_per_seg]
                y = y[
                    spec_start
                    * self.output_hop_size : (spec_start + frames_per_seg)
                    * self.output_hop_size,
                ]
            else:
                x = torch.nn.functional.pad(
                    x, (0, frames_per_seg - x.size(1)), "constant"
                )
                y_mel = torch.nn.functional.pad(
                    y_mel,
                    (0, frames_per_seg - y_mel.size(1)),
                    "constant",
                )
                y = torch.nn.functional.pad(
                    y, (0, self.segment_size - y.size(0)), "constant"
                )
        return (x, y, self.audio_files[index]["basename"], y_mel)

    def __len__(self):
        return len(self.audio_files)

    def get_labels(self):
        return [x["label"] for x in self.audio_files]


class HiFiGANDataModule(BaseDataModule):
    def __init__(self, config: BaseConfig):
        super().__init__(config=config)
        self.use_weighted_sampler = config["training"]["vocoder"][
            "use_weighted_sampler"
        ]

    def load_dataset(self):
        self.dataset = self.config["training"]["vocoder"]["filelist_loader"](
            self.config["training"]["vocoder"]["filelist"]
        )

    def load_train_dataset(self):
        self.train_dataset = SpecDataset(config=self.config, use_segments=True)

    def load_val_dataset(self):
        self.val_dataset = SpecDataset(config=self.config, use_segments=False)


class HiFiGANFineTuneDataModule(BaseDataModule):
    def load_dataset(self):
        self.dataset = SpecDataset(config=self.config, finetune=True)


class FeaturePredictionDataModule(BaseDataModule):
    # TODO: look into compatibility with base data module; ie pin_memory, drop_last, etc
    def load_dataset(self):
        pass


class E2EDataModule(BaseDataModule):
    def load_dataset(self):
        pass
