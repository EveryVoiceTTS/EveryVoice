from functools import partial
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

from everyvoice.config.type_definitions import (
    DatasetTextRepresentation,
    TargetTrainingTextRepresentationLevel,
)
from everyvoice.dataloader import BaseDataModule
from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.dataset import (
    FastSpeech2DataModule,
)
from everyvoice.text.lookups import LookupTables
from everyvoice.text.text_processor import TextProcessor
from everyvoice.utils import (
    _flatten,
    check_dataset_size,
    filter_dataset_based_on_target_text_representation_level,
)
from everyvoice.utils.heavy import get_segments


class E2EDataset(Dataset):
    def __init__(self, dataset, config: EveryVoiceConfig, use_segments=True):
        self.dataset = dataset
        self.config = config
        self.use_segments = use_segments
        self.lookup = LookupTables(config)
        self.sep = "--"
        self.text_processor = TextProcessor(config.feature_prediction.text)
        self.preprocessed_dir = Path(
            self.config.feature_prediction.preprocessing.save_dir
        )
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
            self.config.vocoder.preprocessing.audio.fft_hop_size
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

    def _load_audio(self, bn, spk, lang, dir, fn):
        audio, _ = torchaudio.load(
            str(self.preprocessed_dir / dir / self.sep.join([bn, spk, lang, fn]))
        )
        return audio

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
        if self.config.feature_prediction.model.learn_alignment:
            match self.config.feature_prediction.model.target_text_representation_level:
                case TargetTrainingTextRepresentationLevel.characters:
                    duration = self._load_file(
                        basename,
                        speaker,
                        language,
                        "attn",
                        f"{DatasetTextRepresentation.characters.value}-attn-prior.pt",
                    )
                case TargetTrainingTextRepresentationLevel.ipa_phones | TargetTrainingTextRepresentationLevel.phonological_features:
                    duration = self._load_file(
                        basename,
                        speaker,
                        language,
                        "attn",
                        f"{DatasetTextRepresentation.ipa_phones.value}-attn-prior.pt",
                    )
                case _:
                    raise NotImplementedError(
                        f"{self.config.feature_prediction.model.target_text_representation_level} have not yet been implemented."
                    )
        else:
            duration = self._load_file(
                basename, speaker, language, "duration", "duration.pt"
            )
        match self.config.feature_prediction.model.target_text_representation_level:
            case TargetTrainingTextRepresentationLevel.characters:
                text = torch.IntTensor(
                    self.text_processor.encode_escaped_string_sequence(
                        item["character_tokens"]
                    )
                )
            case TargetTrainingTextRepresentationLevel.ipa_phones | TargetTrainingTextRepresentationLevel.phonological_features:
                text = torch.IntTensor(
                    self.text_processor.encode_escaped_string_sequence(
                        item["phone_tokens"]
                    )
                )
            case _:
                raise NotImplementedError(
                    f"{self.config.feature_prediction.model.target_text_representation_level} have not yet been implemented."
                )
        if TargetTrainingTextRepresentationLevel.characters.value in item:
            raw_text = item[TargetTrainingTextRepresentationLevel.characters.value]
        else:
            raw_text = item.get(
                TargetTrainingTextRepresentationLevel.ipa_phones.value, "text"
            )
        pfs = None
        if (
            self.config.feature_prediction.model.target_text_representation_level
            == TargetTrainingTextRepresentationLevel.phonological_features
        ):
            pfs = self._load_file(basename, speaker, language, "pfs", "pfs.pt")

        energy = self._load_file(basename, speaker, language, "energy", "energy.pt")
        pitch = self._load_file(basename, speaker, language, "pitch", "pitch.pt")
        audio = self._load_audio(
            basename,
            speaker,
            language,
            "audio",
            f"audio-{self.output_sampling_rate}.wav",
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
            # "label": item["label"],
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
        self.collate_fn = partial(
            FastSpeech2DataModule.collate_method,
            learn_alignment=config.feature_prediction.model.learn_alignment,
        )
        self.use_weighted_sampler = (
            config.feature_prediction.training.use_weighted_sampler
        )
        self.batch_size = (
            config.training.batch_size
        )  # TODO: should this be set somewhere else?
        self.load_dataset()
        self.dataset_length = len(self.train_dataset) + len(self.val_dataset)

    def load_dataset(self):
        self.train_dataset = self.config.training.filelist_loader(
            self.config.training.training_filelist
        )
        self.val_dataset = self.config.training.filelist_loader(
            self.config.training.validation_filelist
        )

    def prepare_data(self):
        (
            self.train_dataset,
            self.val_dataset,
        ) = filter_dataset_based_on_target_text_representation_level(
            self.config.feature_prediction.model.target_text_representation_level,
            self.train_dataset,
            self.val_dataset,
            self.batch_size,
        )
        train_samples = len(self.train_dataset)
        val_samples = len(self.val_dataset)
        check_dataset_size(self.batch_size, train_samples, "training")
        check_dataset_size(self.batch_size, val_samples, "validation")
        self.train_dataset = E2EDataset(self.train_dataset, self.config)
        self.val_dataset = E2EDataset(self.val_dataset, self.config)
        # save it to disk
        torch.save(self.train_dataset, self.train_path)
        torch.save(self.val_dataset, self.val_path)
