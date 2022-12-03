""" Preprocessor Module that given a filelist containing text, wav, textgrid:
    - extracts log Mel spectral features
    - extracts pitch (phone-level or frame-level)
    - extracts energy
    - extracts inputs (ex. phonological feats)
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from loguru import logger
from pathos.multiprocessing import ProcessingPool as Pool
from pydantic import BaseModel
from tabulate import tabulate
from torch import Tensor, linalg, mean, tensor
from torchaudio import load as load_audio
from torchaudio import save as save_audio
from torchaudio.functional import compute_kaldi_pitch, resample
from torchaudio.sox_effects import apply_effects_tensor
from tqdm import tqdm

from smts.config import ConfigError
from smts.config.preprocessing_config import Dataset
from smts.model.aligner.config import AlignerConfig
from smts.model.feature_prediction.config import FeaturePredictionConfig
from smts.model.vocoder.config import VocoderConfig
from smts.text import TextProcessor
from smts.utils import write_filelist
from smts.utils.heavy import dynamic_range_compression_torch, get_spectral_transform


class ProgressBar:
    def __init__(self, max_value, disable=False):
        self.max_value = max_value
        self.disable = disable
        self.p = self.pbar()

    def pbar(self):
        return tqdm(total=self.max_value, desc="Loading: ", disable=self.disable)

    def update(self, update_value):
        self.p.update(update_value)

    def close(self):
        self.p.close()


class Scaler:
    def __init__(self):
        self._data = []
        self._tensor_data = None
        self.min = None
        self.max = None
        self.std = None
        self.mean = None
        self.norm_min = None
        self.norm_max = None

    def __len__(self):
        return len(self.data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        raise ValueError(
            f"Sorry, you tried to change the data to {value} but it cannot be changed directly. Either Scaler.append(data), or Scaler.clear_data()"
        )

    def append(self, value):
        self._data += value

    def clear_data(self):
        """Clear data"""
        self.__init__()

    def normalize(self, data):
        """Remove mean and normalize to unit variance"""
        return (data - self.mean) / self.std

    def denormalize(self, data):
        """Get de-normalized value"""
        return (data * self.std) + self.mean

    def calculate_stats(self):
        if not len(self):
            return
        if self._tensor_data is None:
            self._tensor_data = torch.cat(self.data)
        non_nan_data = self._tensor_data[~torch.isnan(self._tensor_data)]
        self.min = torch.min(non_nan_data)
        self.max = torch.max(non_nan_data)
        self.mean = torch.nanmean(self._tensor_data)
        self.std = torch.std(non_nan_data)
        self.norm_max = self.normalize(self.max)
        self.norm_min = self.normalize(self.min)
        return {
            "sample_size": len(self),
            "norm_min": float(self.norm_min),
            "norm_max": float(self.norm_max),
            "min": float(self.min),
            "max": float(self.max),
            "mean": float(self.mean),
            "std": float(self.std),
        }


class Counters(BaseModel):
    duration: float = 0.0
    nans: List[str] = []
    audio_too_long: List[str] = []
    audio_too_short: List[str] = []
    skipped_processes: int = 0
    missing_files: List[str] = []


class Preprocessor:
    def __init__(
        self, config: Union[AlignerConfig, FeaturePredictionConfig, VocoderConfig]
    ):
        self.config = config
        self.pitch_scaler = Scaler()
        self.energy_scaler = Scaler()
        self.datasets = config.preprocessing.source_data
        self.save_dir = Path(config.preprocessing.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.counters = Counters()
        self.audio_config = config.preprocessing.audio
        self.sep = config.preprocessing.value_separator
        self.text_processor = (
            None if isinstance(config, VocoderConfig) else TextProcessor(config)
        )
        self.input_sampling_rate = self.audio_config.input_sampling_rate
        self.output_sampling_rate = self.audio_config.output_sampling_rate
        self.sampling_rate_change = (
            self.output_sampling_rate // self.input_sampling_rate
        )
        # Define Spectral Transform
        # Gah, so many ways to do this: https://github.com/CookiePPP/VocoderComparisons/issues/3

        self.input_spectral_transform = get_spectral_transform(
            self.audio_config.spec_type,
            self.audio_config.n_fft,
            self.audio_config.fft_window_frames,
            self.audio_config.fft_hop_frames,
            sample_rate=self.input_sampling_rate,
            n_mels=self.audio_config.n_mels,
            f_min=self.audio_config.f_min,
            f_max=self.audio_config.f_max,
        )
        self.output_spectral_transform = get_spectral_transform(
            self.audio_config.spec_type,
            self.audio_config.n_fft * self.sampling_rate_change,
            self.audio_config.fft_window_frames * self.sampling_rate_change,
            self.audio_config.fft_hop_frames * self.sampling_rate_change,
            sample_rate=self.input_sampling_rate,
            n_mels=self.audio_config.n_mels,
            f_min=self.audio_config.f_min,
            f_max=self.audio_config.f_max,
        )

        if (
            self.input_spectral_transform is None
            or self.output_spectral_transform is None
        ):
            raise ConfigError(
                f"Spectral feature specification {self.audio_config.spec_type} is not supported. Please edit your config file."
            )

    def load_audio_tensor(self, audio_path: Union[Path, str]):
        """Load audio tensor from file

        Args:
            audio_path (str): path to audio file
        """
        try:
            return torch.load(audio_path)
        except FileNotFoundError:
            logger.error("Audio file not found. Please process audio first.")
            exit()

    def process_audio(
        self,
        wav_path: str,
        normalize=True,
        use_effects=False,
        resample_rate=None,
        sox_effects=None,
    ) -> Union[Tuple[Tensor, int], Tuple[None, None]]:
        """Process audio

        Args:
            wav_path (str): path to wav file
            normalize (bool): volume normalization
        Returns:
            [Tensor, int]: audio Tensor, sampling rate
        """

        audio, sr = load_audio(wav_path, normalize=normalize)
        if use_effects and sox_effects:
            audio, sr = apply_effects_tensor(
                audio,
                sr,
                sox_effects,
            )
        if resample_rate is not None and resample_rate != sr:
            audio = resample(audio, sr, resample_rate)
            sr = resample_rate
        if normalize:
            audio /= torch.max(torch.abs(audio))
            audio *= 0.95
        seconds = len(audio[0]) / sr
        if seconds > self.audio_config.max_audio_length:
            logger.warning(f"Audio too long: {wav_path} ({seconds} seconds)")
            self.counters.audio_too_long.append(os.path.basename(wav_path))
            return None, None
        if seconds < self.audio_config.min_audio_length:
            logger.warning(f"Audio too short: {wav_path} ({seconds} seconds)")
            self.counters.audio_too_short.append(os.path.basename(wav_path))
            return None, None
        self.counters.duration += seconds
        audio = audio.squeeze()  # get rid of channels dimension
        return (audio, sr)

    def extract_spectral_features(
        self, audio_tensor: Tensor, transform, normalize=True
    ):
        """Given an audio tensor, extract the log Mel spectral features
        from the given start and end points

        Args:
            audio_tensor (Tensor): Tensor trimmed according
            transform (torchaudio.transforms): transform to apply; use either Preprocessor.input_spectral_transform or Preprocessor.output_spectral_transform
        """
        mel = transform(audio_tensor)
        if normalize:
            mel = dynamic_range_compression_torch(mel)
        return mel

    @staticmethod
    def _interpolate(x):
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        nans, y = nan_helper(x)
        x[nans] = np.interp(y(nans), y(~nans), x[~nans])
        return x

    def extract_pitch(self, audio_tensor: Tensor):
        """Given an audio tensor, extract the pitch

        TODO: consider CWT and Parselmouth

        Comparison with other implementations:
            - ming024 & Christoph Minxhoffer use the pyworld implementation and interpolate along with phone averaging
            - the Lightspeech implementation seems to use pyworld implementation and not interpolate or average
            - Christoph Minxhoffer reported no significant differences with continuous wavelet transform so it is not implemented here

        Args:
            spectral_feature_tensor (Tensor): tensor of spectral features extracted from audio
        """
        if self.config.preprocessing.pitch_type == "pyworld":
            import pyworld as pw  # This isn't a very good place for an import,

            # but also pyworld is very annoying to install so this is a compromise
            pitch, t = pw.dio(
                audio_tensor.squeeze(0)
                .numpy()
                .astype(
                    np.float64
                ),  # TODO: why are these np.float64, maybe it's just what pw expects?
                self.input_sampling_rate,
                frame_period=self.audio_config.fft_hop_frames
                / self.input_sampling_rate
                * 1000,
                speed=4,
            )
            pitch = pw.stonemask(
                audio_tensor.squeeze(0).numpy().astype(np.float64),
                pitch,
                t,
                self.input_sampling_rate,
            )
            pitch[pitch == 0] = np.nan
            pitch = self._interpolate(pitch)
            pitch = tensor(pitch).float()
        elif self.config.preprocessing.pitch_type == "kaldi":
            pitch = compute_kaldi_pitch(
                waveform=audio_tensor,
                sample_rate=self.input_sampling_rate,
                frame_length=self.audio_config.fft_window_frames
                / self.input_sampling_rate
                * 1000,
                frame_shift=self.audio_config.fft_hop_frames
                / self.input_sampling_rate
                * 1000,
                min_f0=50,
                max_f0=400,
            )[0][
                ..., 1
            ]  # TODO: the docs and C Minxhoffer implementation take [..., 0] but this doesn't appear to be the pitch, at least for this version of torchaudio.
        else:
            raise ConfigError(
                f"Sorry, the pitch estimation type '{self.config.preprocessing.pitch_type}' is not supported. Please edit your config file."
            )
        return pitch

    def average_data_by_durations(self, data, durations):
        current_frame_position = 0
        new_data = []
        for duration in durations.numpy().tolist():
            if duration > 0:
                new_data.append(
                    mean(
                        data[current_frame_position : current_frame_position + duration]
                    )
                )
            else:
                new_data.append(1e-7)
            current_frame_position += duration
        return tensor(new_data)

    def extract_energy(self, spectral_feature_tensor: Tensor):
        """Given a spectral feature tensor, and durations extract the energy averaged across a phone

        Args:
            spectral_feature_tensor (Tensor): tensor of spectral features extracted from audio
            durations (_type_): _descriptiont    #TODO
        """
        return linalg.norm(spectral_feature_tensor, dim=0)

    def extract_text_inputs(self, text, use_pfs=False) -> Tensor:
        """Given some text, normalize it, g2p it, and save as one-hot or multi-hot phonological feature vectors

        Args:
            text (str): text
        """
        if self.text_processor is None:
            raise ValueError("Text processor not initialized")
        if use_pfs:
            return torch.Tensor(
                self.text_processor.text_to_phonological_features(text)
            ).long()
        else:
            return torch.Tensor(self.text_processor.text_to_sequence(text)).long()

    def print_duration(self):
        """Convert seconds to a human readable format"""
        seconds = int(self.counters.duration)
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{hours}h {minutes}m {seconds}s"

    def report(self, processed, tablefmt="simple"):
        """Print a report of the dataset processing"""
        headers = ["type", "quantity"]
        table = [
            ["missing files", len(self.counters.missing_files)],
            [
                "missing symbols",
                len(self.text_processor.missing_symbols) if self.text_processor else 0,
            ],
            [
                "duplicate symbols",
                len(self.text_processor.duplicate_symbols)
                if self.text_processor
                else 0,
            ],
            ["skipped processes", self.counters.skipped_processes],
            ["nans", self.counters.nans],
            ["audio_too_short", len(self.counters.audio_too_short)],
            ["audio_too_long", len(self.counters.audio_too_long)],
            ["duration", self.print_duration()],
        ]
        return tabulate(table, headers, tablefmt=tablefmt)

    def _preprocess_audio_or_dependent(
        self,
        dataset_info,
        item,
        speaker,
        language,
        overwrite,
        process_audio,
        process_energy,
        process_pitch,
        process_spec,
    ):
        """Process Audio or Dependents (Spectral Features, Pitch, and Energy)"""
        input_audio_save_path = (
            self.save_dir
            / "audio"
            / self.sep.join(
                [
                    item["basename"],
                    speaker,
                    language,
                    f"audio-{self.input_sampling_rate}.pt",
                ]
            )
        )
        output_audio_save_path = (
            self.save_dir
            / "audio"
            / self.sep.join(
                [
                    item["basename"],
                    speaker,
                    language,
                    f"audio-{self.output_sampling_rate}.pt",
                ]
            )
        )
        audio_paths_exist = (
            input_audio_save_path.exists() or output_audio_save_path.exists()
        )
        if not overwrite and audio_paths_exist:
            self.counters.skipped_processes += 1
        else:
            audio, _ = self.process_audio(
                dataset_info.data_dir / (item["basename"] + ".wav"),
                resample_rate=self.input_sampling_rate,
            )
            if audio is None:
                return None  # Skip processing audio and dependents if didn't pass process_audio
            if process_audio:
                torch.save(audio, input_audio_save_path)
            if self.input_sampling_rate != self.output_sampling_rate:
                output_audio, _ = self.process_audio(
                    dataset_info.data_dir / (item["basename"] + ".wav"),
                    resample_rate=self.output_sampling_rate,
                )
                if process_audio:
                    torch.save(output_audio, output_audio_save_path)
            else:
                output_audio = audio
        if process_pitch:
            save_path = (
                self.save_dir
                / "pitch"
                / self.sep.join([item["basename"], speaker, language, "pitch.pt"])
            )
            if not overwrite and save_path.exists():
                self.counters.skipped_processes += 1
            else:
                pitch = self.extract_pitch(audio)
                if (
                    isinstance(self.config, FeaturePredictionConfig)
                    and self.config.model.variance_adaptor.variance_predictors.pitch.level
                    == "phone"
                ):
                    dur_path = (
                        self.save_dir
                        / "duration"
                        / self.sep.join(
                            [item["basename"], speaker, language, "duration.pt"]
                        )
                    )
                    durs = torch.load(dur_path)
                    pitch = self.average_data_by_durations(pitch, durs)
                self.pitch_scaler.data.append(pitch)
                torch.save(
                    pitch,
                    save_path,
                )
        if process_spec or process_energy:
            self._preprocess_spec_or_dependent(
                audio,
                output_audio,
                item,
                speaker,
                language,
                overwrite,
                process_energy,
                process_spec,
            )

    def _preprocess_spec_or_dependent(
        self,
        audio,
        output_audio,
        item,
        speaker,
        language,
        overwrite,
        process_energy,
        process_spec,
    ):
        """Process Spectral Features or Dependents (Energy)"""
        input_spec_save_path = (
            self.save_dir
            / "spec"
            / self.sep.join(
                [
                    item["basename"],
                    speaker,
                    language,
                    f"spec-{self.input_sampling_rate}-{self.audio_config.spec_type}.pt",
                ]
            )
        )
        output_spec_save_path = (
            self.save_dir
            / "spec"
            / self.sep.join(
                [
                    item["basename"],
                    speaker,
                    language,
                    f"spec-{self.output_sampling_rate}-{self.audio_config.spec_type}.pt",
                ]
            )
        )
        if not overwrite and (
            input_spec_save_path.exists() or output_spec_save_path.exists()
        ):
            self.counters.skipped_processes += 1
        else:
            spec = self.extract_spectral_features(audio, self.input_spectral_transform)
            if process_spec:
                torch.save(spec, input_spec_save_path)
                if self.input_sampling_rate != self.output_sampling_rate:
                    output_spec = self.extract_spectral_features(
                        output_audio, self.output_spectral_transform
                    )
                    torch.save(output_spec, output_spec_save_path)
        if process_energy:
            save_path = (
                self.save_dir
                / "energy"
                / self.sep.join([item["basename"], speaker, language, "energy.pt"])
            )
            if not overwrite and save_path.exists():
                self.counters.skipped_processes += 1
            else:
                energy = self.extract_energy(spec)
                if (
                    self.config.model.variance_adaptor.variance_predictors.energy.level
                    == "phone"
                ):
                    dur_path = (
                        self.save_dir
                        / "duration"
                        / self.sep.join(
                            [item["basename"], speaker, language, "duration.pt"]
                        )
                    )
                    durs = torch.load(dur_path)
                    energy = self.average_data_by_durations(energy, durs)
                self.energy_scaler.data.append(energy)
                torch.save(energy, save_path)

    def _preprocess_item(self, item, kwargs):  # noqa: C901
        """Text, Phonological features, SoX Audio can all be processed independently, but Spectral features and Pitch
        depend on Audio first being processed, and energy depends on Spectral features first being processed. So,
        to simplify, processing is separated into stages."""
        speaker = "default" if "speaker" not in item else item["speaker"]
        language = "default" if "language" not in item else item["language"]
        item = {**item, **{"speaker": speaker, "language": language}}
        kwargs["dataset_info"].data_dir = Path(kwargs["dataset_info"].data_dir)
        if kwargs.get("process_text"):
            save_path = (
                self.save_dir
                / "text"
                / self.sep.join([item["basename"], speaker, language, "text.pt"])
            )
            if not kwargs.get("overwrite") and save_path.exists():
                self.counters.skipped_processes += 1
            else:
                torch.save(
                    self.extract_text_inputs(item["text"]),
                    save_path,
                )
                item["raw_text"] = item["text"]
                item["clean_text"] = self.text_processor.clean_text(item["text"])
        if kwargs.get("process_pfs"):
            save_path = (
                self.save_dir
                / "text"
                / self.sep.join([item["basename"], speaker, language, "pfs.pt"])
            )
            if not kwargs.get("overwrite") and save_path.exists():
                self.counters.skipped_processes += 1
            else:
                torch.save(
                    self.extract_text_inputs(item["text"], use_pfs=True), save_path
                )
        if kwargs.get("process_sox_audio"):
            save_path = (
                self.save_dir
                / "audio"
                / self.sep.join([item["basename"], speaker, language, "processed.wav"])
            )
            if not kwargs.get("overwrite") and save_path.exists():
                self.counters.skipped_processes += 1
            else:
                save_audio(
                    save_path,
                    self.process_audio(
                        kwargs["dataset_info"].data_dir / (item["basename"] + ".wav"),
                        use_effects=True,
                        sox_effects=kwargs["dataset_info.sox_effects"],
                    ),
                    self.audio_config.alignment_sampling_rate,
                    encoding="PCM_S",
                    bits_per_sample=self.audio_config.alignment_bit_depth,
                )
        if any(
            [
                kwargs.get("process_audio"),
                kwargs.get("process_energy"),
                kwargs.get("process_pitch"),
                kwargs.get("process_spec"),
            ]
        ):
            self._preprocess_audio_or_dependent(
                kwargs["dataset_info"],
                item,
                speaker,
                language,
                kwargs.get("overwrite", False),
                kwargs.get("process_audio", False),
                kwargs.get("process_energy", False),
                kwargs.get("process_pitch", False),
                kwargs.get("process_spec", False),
            )
        if isinstance(kwargs["dataset_info"].label, str):
            item["label"] = kwargs["dataset_info"].label
        elif callable(kwargs["dataset_info"]["label"]):
            item["label"] = kwargs["dataset_info"].label(item)
        else:
            raise ValueError(
                f"Label for dataset '{kwargs['dataset_info'].label}' is neither a string nor a callable."
            )
        return item

    def _create_stats_batch(self, x):
        mel_val = x["mel"]
        result = {"mel": {"mean": torch.nanmean(mel_val)}}
        result["mel"]["std"] = torch.std(mel_val[~torch.isnan(mel_val)])
        pitch = x["pitch"]
        energy = x["energy"]
        result["pitch"] = {
            "min": torch.min(pitch[~torch.isnan(pitch)]),
            "max": torch.max(pitch[~torch.isnan(pitch)]),
            "mean": torch.nanmean(pitch),
            "std": torch.std(pitch[~torch.isnan(pitch)]),
        }
        result["energy"] = {
            "min": torch.min(energy[~torch.isnan(energy)]),
            "max": torch.max(energy[~torch.isnan(energy)]),
            "mean": torch.nanmean(energy),
            "std": torch.std(energy[~torch.isnan(energy)]),
        }
        return result

    def normalize(self, energy=False, pitch=False, overwrite=True):
        """Normalize pitch and energy to unit variance"""
        logger.info("Scaling dataset statistics...")
        stats = {}
        if energy:
            energy_stats = self.energy_scaler.calculate_stats()
            for path in tqdm(
                (self.config.preprocessing.save_dir / "energy").glob("*energy*"),
                desc="Normalizing energy values",
            ):
                energy = torch.load(path)
                energy = self.energy_scaler.normalize(energy)
                torch.save(energy, path)
            stats["energy"] = energy_stats
        if pitch:
            pitch_stats = self.pitch_scaler.calculate_stats()
            for path in tqdm(
                (self.config.preprocessing.save_dir / "pitch").glob("*pitch*"),
                desc="Normalizing pitch values",
            ):
                pitch = torch.load(path)
                pitch = self.pitch_scaler.normalize(pitch)
                torch.save(pitch, path)
            stats["pitch"] = pitch_stats
        return stats

    def _collect_files_from_filelist(self, filelist, data_dir):
        for f in filelist:
            audio_path = data_dir / (f["basename"] + ".wav")
            if not audio_path.exists():
                logger.warning(f"File '{f}' is missing and will not be processed.")
                self.counters.missing_files.append(f)
            else:
                yield f

    def preprocess(  # noqa: C901
        self,
        output_path="processed_filelist.psv",
        compute_stats=False,
        cpus=mp.cpu_count(),
        **kwargs,
    ):
        if kwargs.get("process_audio"):
            (self.save_dir / "audio").mkdir(parents=True, exist_ok=True)
        if kwargs.get("process_energy"):
            (self.save_dir / "energy").mkdir(parents=True, exist_ok=True)
        if kwargs.get("process_pitch"):
            (self.save_dir / "pitch").mkdir(parents=True, exist_ok=True)
        if kwargs.get("process_spec"):
            (self.save_dir / "spec").mkdir(parents=True, exist_ok=True)
        if kwargs.get("process_text"):
            (self.save_dir / "text").mkdir(parents=True, exist_ok=True)
        write_path = self.save_dir / output_path
        if write_path.exists() and not kwargs["overwrite"]:
            logger.error(
                f"Preprocessed filelist at '{write_path}' already exists. Please either set overwrite=True or choose a new path"
            )
            exit()
        processed = 0
        files = []
        # Sanity check
        for dataset in self.datasets:
            data_dir = Path(dataset.data_dir)
            if not data_dir.exists():
                logger.error(
                    f"Data directory '{data_dir}' does not exist. Please check your config file."
                )
                exit()
            # TODO: more sanity checks
        # Actual processing
        dataset: Dataset
        process_energy = kwargs.get("process_energy")
        process_pitch = kwargs.get("process_pitch")
        for dataset in tqdm(self.datasets, total=len(self.datasets)):
            data_dir = Path(dataset.data_dir)
            filelist = dataset.filelist_loader(dataset.filelist)
            logger.info(f"Collecting files for {dataset.label}")

            list_len = len(filelist)
            all_valid_files = list(
                self._collect_files_from_filelist(filelist, data_dir)
            )
            valid_len = len(all_valid_files)
            chunk_size = int(valid_len / cpus)
            if chunk_size == 0 or cpus < 2:
                kwargs = {**{"dataset_info": dataset}, **kwargs}
                for item in tqdm(all_valid_files):
                    files.append(self._preprocess_item(item, kwargs))
            else:
                pool = Pool(nodes=cpus)
                logger.info(
                    f"Processing {list_len} files in {chunk_size} file chunks on {cpus} CPUs"
                )
                chunks = [
                    all_valid_files[i : i + chunk_size]
                    for i in range(0, valid_len, chunk_size)
                ]
                kwargs_iterable = [
                    {
                        **{
                            "dataset_info": dataset,
                            "pbar": i < min(5, cpus),
                            "position": i + 1,
                        },
                        **kwargs,
                    }
                    for i, _ in enumerate(range(len(chunks)))
                ]

                def _preprocess_chunk(chunk, kwargs):
                    items = []
                    if kwargs["pbar"]:
                        for item in tqdm(
                            chunk,
                            position=kwargs["position"],
                            desc=f"Subprocess #{kwargs['position']} progress",
                        ):
                            items.append(self._preprocess_item(item, kwargs))
                    else:
                        for item in chunk:
                            items.append(self._preprocess_item(item, kwargs))
                    return items, self.counters, self.energy_scaler, self.pitch_scaler

                for items, counters, e_scaler, p_scaler in tqdm(
                    pool.uimap(_preprocess_chunk, chunks, kwargs_iterable),
                    position=0,
                    desc="Chunk processing progress",
                    total=len(chunks),
                ):
                    files += items
                    self.counters.duration += counters.duration
                    self.counters.nans += counters.nans
                    self.counters.audio_too_long += counters.audio_too_long
                    self.counters.audio_too_short += counters.audio_too_short
                    self.counters.skipped_processes += counters.skipped_processes
                    if process_energy:
                        self.energy_scaler.data.extend(e_scaler.data)
                    if process_pitch:
                        self.pitch_scaler.data.extend(p_scaler.data)
                    processed += len(items)

        logger.info(self.report(processed))
        write_filelist(files, self.save_dir / output_path)
        if compute_stats:
            logger.info(
                "Normalizing pitch and energy to unit variance and calculating stats"
            )
            stat_path = self.save_dir / "stats.json"
            stats = self.normalize(
                energy=process_energy,
                pitch=process_pitch,
                overwrite=kwargs["overwrite"],
            )
            if stat_path.exists():
                with open(stat_path, "r", encoding="utf8") as f:
                    old_stats = json.load(f)
                stats = {**old_stats, **stats}
            with open(stat_path, "w", encoding="utf8") as f:
                json.dump(stats, f)
