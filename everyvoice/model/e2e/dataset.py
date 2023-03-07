import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, random_split

from everyvoice.dataloader import BaseDataModule
from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.text import TextProcessor
from everyvoice.text.lookups import LookupTables
from everyvoice.utils import check_dataset_size
from everyvoice.utils.heavy import _flatten, get_segments


class E2EDataset(Dataset):
    def __init__(self, dataset, config: EveryVoiceConfig, use_segments=True):
        self.dataset = dataset
        self.config = config
        self.use_segments = use_segments
        self.lookup = LookupTables(config)
        self.sep = config.feature_prediction.preprocessing.value_separator
        self.text_processor = TextProcessor(config.feature_prediction)
        self.preprocessed_dir = Path(
            self.config.feature_prediction.preprocessing.save_dir
        )
        random.seed(self.config.training.seed)
        self.output_sampling_rate = (
            self.config.vocoder.preprocessing.audio.output_sampling_rate
        )
        self.input_sampling_rate = (
            self.config.feature_prediction.preprocessing.audio.input_sampling_rate
        )
        self.sampling_rate_change = (
            self.output_sampling_rate // self.input_sampling_rate
        )
        self.output_hop_size = (
            self.config.vocoder.preprocessing.audio.fft_hop_frames
            * self.sampling_rate_change
        )
        self.frame_segment_size = (
            self.config.vocoder.preprocessing.audio.vocoder_segment_size
            // self.output_hop_size
        )
        self.speaker2id = self.lookup.speaker2id
        self.lang2id = self.lookup.lang2id

    def _load_file(self, bn, spk, lang, dir, fn):
        return torch.load(
            self.preprocessed_dir / dir / self.sep.join([bn, spk, lang, fn])
        )

    def __getitem__(self, index):
        """
        Returns dict with keys: {
            "mel"
            "duration"
            "pfs"
            "text"
            "raw_text"
            "basename"
            "speaker"
            "speaker_id"
            "language"
            "language_id"
            "label"
            "energy"
            "pitch"
            "audio"
            "audio_mel"
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
            "spec",
            f"spec-{self.input_sampling_rate}-{self.config.feature_prediction.preprocessing.audio.spec_type}.pt",
        ).transpose(
            0, 1
        )  # [mel_bins, frames] -> [frames, mel_bins]
        duration = self._load_file(
            basename, speaker, language, "duration", "duration.pt"
        )
        text = self._load_file(basename, speaker, language, "text", "text.pt")
        raw_text = item["raw_text"]
        pfs = None
        if self.config.feature_prediction.model.use_phonological_feats:
            pfs = self._load_file(basename, speaker, language, "text", "pfs.pt")

        energy = self._load_file(basename, speaker, language, "energy", "energy.pt")
        pitch = self._load_file(basename, speaker, language, "pitch", "pitch.pt")
        audio = self._load_file(
            basename,
            speaker,
            language,
            "audio",
            f"audio-{self.output_sampling_rate}.pt",
        )
        if audio.dim() == 1:
            audio = audio.unsqueeze(
                0
            )  # audio should have channel dimension for vocoder
        audio_mel = self._load_file(
            basename,
            speaker,
            language,
            "spec",
            f"spec-{self.output_sampling_rate}-{self.config.vocoder.preprocessing.audio.spec_type}.pt",
        )  # [mel_bins, frames]
        if self.use_segments:
            audio, start_sample = get_segments(
                audio, self.config.vocoder.preprocessing.audio.vocoder_segment_size
            )
            audio_mel, start_frame = get_segments(
                audio_mel, self.frame_segment_size, start_sample // self.output_hop_size
            )
            input_start_frame = (
                start_frame // self.sampling_rate_change
            )  # we need to take the corresponding segment from the output of the feature prediction network, so this stores the index to start from
        return {
            "mel": mel,
            "duration": duration,
            "pfs": pfs,
            "text": text,
            "raw_text": raw_text,
            "basename": basename,
            "speaker": speaker,
            "speaker_id": speaker_id,
            "language": language,
            "language_id": language_id,
            "label": item["label"],
            "energy": energy,
            "pitch": pitch,
            "audio": audio,
            "audio_mel": audio_mel,
            "segment_start_frame": input_start_frame if self.use_segments else None,
        }

    def __len__(self):
        return len(self.dataset)

    def get_labels(self):
        return [x["label"] for x in self.dataset]


class E2EDataModule(BaseDataModule):
    def __init__(self, config: EveryVoiceConfig):
        super().__init__(config=config)
        self.collate_fn = self.collate_method
        self.use_weighted_sampler = (
            config.feature_prediction.training.use_weighted_sampler
        )
        self.batch_size = (
            config.training.batch_size
        )  # TODO: should this be set somewhere else?
        self.train_split = config.training.train_split
        self.load_dataset()
        self.dataset_length = len(self.dataset)

    @staticmethod
    def collate_method(data):
        data = [_flatten(x) for x in data]
        data = {k: [dic[k] for dic in data] for k in data[0]}
        text_lens = torch.LongTensor([text.size(0) for text in data["text"]])
        mel_lens = torch.LongTensor([mel.size(0) for mel in data["mel"]])
        max_mel = max(mel_lens)
        max_text = max(text_lens)
        for key in data:
            if isinstance(data[key][0], np.ndarray):
                data[key] = [torch.tensor(x) for x in data[key]]
            if torch.is_tensor(data[key][0]):
                data[key] = pad_sequence(data[key], batch_first=True, padding_value=0)
            if isinstance(data[key][0], int):
                data[key] = torch.tensor(data[key]).long()
        data["src_lens"] = text_lens
        data["mel_lens"] = mel_lens
        data["max_src_len"] = max_text
        data["max_mel_len"] = max_mel
        return data

    def load_dataset(self):
        self.dataset = self.config.training.filelist_loader(
            self.config.training.filelist
        )

    def prepare_data(self):
        train_samples = int(self.dataset_length * self.train_split)
        val_samples = self.dataset_length - train_samples
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_samples, val_samples]
        )
        check_dataset_size(self.batch_size, train_samples, "training")
        check_dataset_size(self.batch_size, val_samples, "validation")
        self.train_dataset = E2EDataset(self.train_dataset, self.config)
        self.val_dataset = E2EDataset(self.val_dataset, self.config)
        # save it to disk
        torch.save(self.train_dataset, self.train_path)
        torch.save(self.val_dataset, self.val_path)
