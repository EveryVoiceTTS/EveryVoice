import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from smts.config.base_config import BaseConfig
from smts.dataloader.imbalanced_sampler import ImbalancedDatasetSampler
from smts.text import TextProcessor
from smts.text.lookups import LookupTables
from smts.utils import collate_fn, expand


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
            num_workers=self.config["training"]["train_data_workers"],
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
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
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
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


class FastSpeechDataset(Dataset):
    """
    To debug, set num_workers=0 and batch_size=1
    """

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.lookup = LookupTables(config)
        self.sep = config["preprocessing"]["value_separator"]
        self.text_processor = TextProcessor(config)
        self.preprocessed_dir = Path(self.config["preprocessing"]["save_dir"])
        random.seed(self.config["training"]["feature_prediction"]["seed"])
        self.sampling_rate = self.config["preprocessing"]["audio"][
            "input_sampling_rate"
        ]
        self.speaker2id = self.lookup.speaker2id
        self.lang2id = self.lookup.lang2id

    def _load_file(self, bn, spk, lang, fn):
        return torch.load(self.preprocessed_dir / self.sep.join([bn, spk, lang, fn]))

    def __getitem__(self, index):
        """
        Returns dict with keys: {
            "phones",
            "duration",
            "silence_mask",
            "unexpanded_silence_mask",
            "priors",
            "audio",
            "speaker",
            "text",
            "basename",
            "variances",
            "labels",
        }
        """
        item = self.dataset[index]
        speaker = "default" if "speaker" not in item else item["speaker"]
        language = "default" if "language" not in item else item["language"]
        speaker_id = self.speaker2id[speaker]
        language_id = self.lang2id[language]
        basename = item["basename"]
        mel = self._load_file(
            basename,
            speaker,
            language,
            f"spec-{self.sampling_rate}-{self.config['preprocessing']['audio']['spec_type']}.npy",
        ).transpose(
            0, 1
        )  # [mel_bins, frames] -> [frames, mel_bins]
        duration = self._load_file(basename, speaker, language, "duration.npy")
        text = self._load_file(basename, speaker, language, "text.npy")
        raw_text = item["raw_text"]
        pfs = None
        if self.config["model"]["transformer"]["use_phon_feats"]:
            pfs = self._load_file(basename, speaker, language, "pfs.npy")
        variances = {
            key: self._load_file(basename, speaker, language, f"{key}.npy")
            for key in self.config["model"]["variance_adaptor"]["variances"]
        }

        # TODO: Fix text processor and resolve how to deal with punctuation and
        # potential mismatch between duration/textgrid and text
        # DONE: This was resolved by the use of an aligner that uses the same text processor
        # TODO: silence masks
        # DONE: There won't really be any silences if we aren't using MFA, this might be a problem
        # Durations & Silence Mask
        silence_ids = [
            self.text_processor.cleaned_text_to_sequence(x)[0]
            for x in self.config["text"]["symbols"]["silence"]
        ]
        silence_masks = [np.array(text) == s for s in silence_ids]
        unexpanded_silence_mask = np.logical_or.reduce(silence_masks)
        silence_mask = expand(unexpanded_silence_mask, duration)

        # Priors

        return {
            "mel": mel,
            "duration": duration,
            "silence_mask": silence_mask,
            "unexpanded_silence_mask": unexpanded_silence_mask,
            "pfs": pfs,
            "text": text,
            "raw_text": raw_text,
            "speaker": speaker,
            "speaker_id": speaker_id,
            "language": language,
            "language_id": language_id,
            "label": item["label"],
            "variances": variances,
        }

    def __len__(self):
        return len(self.dataset)

    def get_labels(self):
        return [x["label"] for x in self.dataset]


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
        self.batch_size = config["training"]["vocoder"]["batch_size"]
        self.train_split = self.config["training"]["vocoder"]["train_split"]

    def load_dataset(self):
        self.dataset = self.config["training"]["vocoder"]["filelist_loader"](
            self.config["training"]["vocoder"]["filelist"]
        )

    def prepare_data(self):
        self.load_dataset()
        train_split = int(len(self.dataset) * self.train_split)
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


class HiFiGANFineTuneDataModule(BaseDataModule):
    def __init__(self, config: BaseConfig):
        super().__init__(config=config)
        self.use_weighted_sampler = config["training"]["vocoder"][
            "use_weighted_sampler"
        ]
        self.batch_size = config["training"]["vocoder"]["batch_size"]
        self.train_split = self.config["training"]["vocoder"]["train_split"]

    def load_dataset(self):
        self.dataset = SpecDataset(config=self.config, finetune=True)


class FeaturePredictionDataModule(BaseDataModule):
    # TODO: look into compatibility with base data module; ie pin_memory, drop_last, etc
    def __init__(self, config: BaseConfig):
        super().__init__(config=config)
        self.use_weighted_sampler = config["training"]["feature_prediction"][
            "use_weighted_sampler"
        ]
        self.batch_size = config["training"]["feature_prediction"]["batch_size"]
        self.train_split = self.config["training"]["feature_prediction"]["train_split"]
        self.load_dataset()

    def load_dataset(self):
        self.dataset = self.config["training"]["feature_prediction"]["filelist_loader"](
            self.config["training"]["feature_prediction"]["filelist"]
        )

    def prepare_data(self):
        train_split = int(len(self.dataset) * self.train_split)

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_split, len(self.dataset) - train_split]
        )
        self.train_dataset = FastSpeechDataset(self.train_dataset, self.config)
        self.val_dataset = FastSpeechDataset(self.val_dataset, self.config)
        # save it to disk
        torch.save(self.train_dataset, self.train_path)
        torch.save(self.val_dataset, self.val_path)


class E2EDataModule(BaseDataModule):
    def load_dataset(self):
        pass
