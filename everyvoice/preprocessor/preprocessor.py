""" Preprocessor Module that given a filelist containing text, wav, textgrid:
    - extracts log Mel spectral features
    - extracts pitch (phone-level or frame-level)
    - extracts energy
    - extracts inputs (ex. phonological feats)
"""

import functools
import multiprocessing as mp
import random
import sys
from collections import Counter
from glob import glob
from multiprocessing import Manager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from clipdetect import detect_clipping
from joblib import Parallel, delayed
from loguru import logger
from rich import print as rich_print
from rich.panel import Panel
from rich.style import Style
from tabulate import tabulate
from torchaudio.functional import resample
from torchaudio.sox_effects import apply_effects_tensor
from tqdm import tqdm

from everyvoice.exceptions import ConfigError
from everyvoice.model.aligner.config import AlignerConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.preprocessor.attention_prior import BetaBinomialInterpolator
from everyvoice.preprocessor.helpers import Counters, Scaler, save_tensor
from everyvoice.text import TextProcessor
from everyvoice.utils import (
    generic_dict_loader,
    n_times,
    tqdm_joblib_context,
    write_filelist,
)
from everyvoice.utils.heavy import (
    dynamic_range_compression_torch,
    get_spectral_transform,
)


class Preprocessor:
    def __init__(self, config: AlignerConfig | FeaturePredictionConfig | VocoderConfig):
        self.config = config
        self.counters = Counters(Manager())
        self.cpus = 0
        self.pitch_scaler = Scaler()
        self.energy_scaler = Scaler()
        self.datasets = config.preprocessing.source_data
        self.save_dir = Path(config.preprocessing.save_dir)
        self.audio_config = config.preprocessing.audio
        self.sep = "--"
        self.text_processor = (
            None if isinstance(config, VocoderConfig) else TextProcessor(config)
        )
        self.overwrite = False
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
            self.audio_config.fft_window_size,
            self.audio_config.fft_hop_size,
            sample_rate=self.input_sampling_rate,
            n_mels=self.audio_config.n_mels,
            f_min=self.audio_config.f_min,
            f_max=self.audio_config.f_max,
        )
        self.output_spectral_transform = get_spectral_transform(
            self.audio_config.spec_type,
            self.audio_config.n_fft * self.sampling_rate_change,
            self.audio_config.fft_window_size * self.sampling_rate_change,
            self.audio_config.fft_hop_size * self.sampling_rate_change,
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
                f"Spectral feature specification '{self.audio_config.spec_type}' is not supported. Please edit your config file."
            )

    def load_audio(self, audio_path: Path) -> tuple[torch.Tensor, int, float]:
        """
        Load an audio file and calculate its duration.

        Args:
            audio_path: path to audio file

        Returns: (audio as a Tensor, sampling rate, duration in seconds)
        """
        audio, sr = torchaudio.load(str(audio_path))
        seconds = len(audio[0]) / sr
        return audio, sr, seconds

    def process_audio(
        self,
        wav_path: Path,
        normalize=True,
        use_effects=True,
        resample_rate=None,
        sox_effects=None,
        save_wave=False,
        update_counters=True,  # unset this when processing the same file a second time
    ) -> tuple[torch.Tensor, int] | tuple[None, None]:
        """Process audio

        Args:
            wav_path (Path): path to wav file
            normalize (bool): volume normalization
        Returns:
            Tensor: (audio as a Tensor, sampling rate)
        """
        audio, sr, seconds = self.load_audio(wav_path)

        if seconds > self.audio_config.max_audio_length:
            logger.warning(
                f"Audio too long: {wav_path} ({seconds} seconds - we will skip this file)"
            )
            if update_counters:
                self.counters.increment("audio_too_long")
            return None, None
        if seconds < self.audio_config.min_audio_length:
            logger.warning(
                f"Audio too short: {wav_path} ({seconds} seconds - we will skip this file)"
            )
            if update_counters:
                self.counters.increment("audio_too_short")
            return None, None

        loudness_transform = torchaudio.transforms.Loudness(sr)
        loudness = loudness_transform(audio)
        if (
            torch.isnan(loudness) or loudness < -36
        ):  # This is a conservative threshold, so some very quiet/silent files may still get through
            logger.warning(f"Audio empty: {wav_path} - we will skip this file")
            if update_counters:
                self.counters.increment("audio_empty")
            return None, None

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

        if update_counters:
            self.counters.increment("processed_files")
            self.counters.increment("duration", seconds)
        if save_wave:
            torchaudio.save(
                str(wav_path) + ".processed.wav",
                audio,
                sr,
                encoding="PCM_S",
                bits_per_sample=self.audio_config.target_bit_depth,
            )
        audio = audio.squeeze()  # get rid of channels dimension
        return audio, sr

    def extract_spectral_features(
        self, audio_tensor: torch.Tensor, transform, normalize=True
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

    def extract_pitch(self, audio_tensor: torch.Tensor):
        """Given an audio tensor, extract the pitch

        TODO: consider CWT and Parselmouth

        Comparison with other implementations:
            - ming024 & Christoph Minxhoffer use the pyworld implementation and interpolate along with phone averaging
            - the Lightspeech implementation seems to use pyworld implementation and not interpolate or average
            - Christoph Minxhoffer reported no significant differences with continuous wavelet transform so it is not implemented here

        Args:
            audio_tensor (Tensor): 1D tensor of audio samples
        """
        import pyworld as pw  # This isn't a very good place for an import,

        # but also pyworld is very annoying to install so this is a compromise
        pitch, t = pw.dio(
            audio_tensor.squeeze(0)
            .numpy()
            .astype(
                np.float64
            ),  # TODO: why are these np.float64, maybe it's just what pw expects?
            self.input_sampling_rate,
            frame_period=self.audio_config.fft_hop_size
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
        try:
            pitch = self._interpolate(pitch)
        except ValueError:
            # TODO: we should warn the user about pitch-less samples
            pitch[np.isnan(pitch)] = 0
        pitch = torch.tensor(pitch).float()
        return pitch

    def average_data_by_durations(self, data, durations):
        current_frame_position = 0
        new_data = []
        for duration in durations.numpy().tolist():
            if duration > 0:
                new_data.append(
                    torch.mean(
                        data[current_frame_position : current_frame_position + duration]
                    )
                )
            else:
                new_data.append(1e-7)
            current_frame_position += duration
        return torch.tensor(new_data)

    def extract_energy(self, spectral_feature_tensor: torch.Tensor):
        """Given a spectral feature tensor, and durations extract the energy averaged across a phone

        Args:
            spectral_feature_tensor (Tensor): tensor of spectral features extracted from audio
            durations (_type_): _descriptiont    #TODO
        """
        return torch.linalg.norm(spectral_feature_tensor, dim=0)

    # This method is static because we want to use it in Datasets without setting the Preprocessor as an attribute
    @staticmethod
    def extract_text_inputs(
        text, text_processor: TextProcessor, use_pfs=False, quiet=False
    ) -> torch.Tensor:
        """
        Given some text and a text_processor, normalize it, g2p it, and save as one-hot or multi-hot phonological feature vectors

        Args:
            text (str): text
            text_processor (TextProcessor): a text processor
            use_pfs:
            quiet: suppress warnings
        """
        if text_processor is None:
            raise ValueError("Text processor not initialized")
        if use_pfs:
            return torch.Tensor(
                text_processor.text_to_phonological_features(text, quiet)
            ).long()
        else:
            return torch.Tensor(text_processor.text_to_sequence(text, quiet)).long()

    def print_duration(self):
        """Convert seconds to a human readable format"""
        seconds = int(self.counters.value("duration"))
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{hours}h {minutes}m {seconds}s"

    def report(self, tablefmt="simple"):
        """Print a report of the dataset processing"""
        headers = ["type", "quantity"]
        table = [
            ["processed files", self.counters.value("processed_files")],
            [
                "previously processed files",
                self.counters.value("previously_processed_files"),
            ],
            ["missing files", self.counters.value("missing_files")],
            [
                "missing symbols",
                len(self.text_processor.missing_symbols) if self.text_processor else 0,
            ],
            [
                "duplicate symbols",
                (
                    len(self.text_processor.duplicate_symbols)
                    if self.text_processor
                    else 0
                ),
            ],
            ["skipped processes", self.counters.value("skipped_processes")],
            ["nans", self.counters.value("nans")],
            ["audio_empty", self.counters.value("audio_empty")],
            ["audio_too_short", self.counters.value("audio_too_short")],
            ["audio_too_long", self.counters.value("audio_too_long")],
            ["duration", self.print_duration()],
        ]
        return tabulate(table, headers, tablefmt=tablefmt)

    def get_speaker_and_language(self, item):
        """Unless the dataset already has values for speaker and language, set them to 'default'"""
        if "speaker" in item and "language" in item:
            return item
        speaker = "default" if "speaker" not in item else item["speaker"]
        language = "default" if "language" not in item else item["language"]
        return {**item, **{"speaker": speaker, "language": language}}

    def compute_stats(
        self, energy=True, pitch=True
    ) -> tuple[Optional[Scaler], Optional[Scaler]]:
        if self.cpus > 1:
            parallel = Parallel(n_jobs=self.cpus, backend="loky", batch_size=500)
        if energy:
            energy_scaler = Scaler()
            # Until Python 3.13, pathlib.Path.glob() doesn't work with symlinks: https://github.com/python/cpython/issues/77609
            paths = glob(
                str(self.config.preprocessing.save_dir / "energy/**/*energy*"),
                recursive=True,
            )
            if self.cpus > 1:
                logger.info("Gathering energy values")
                with tqdm_joblib_context(tqdm(desc="Gathering energy values")):
                    for energy_data in parallel(
                        delayed(torch.load)(path) for path in paths
                    ):
                        energy_scaler.data.append(energy_data)
            else:
                for path in tqdm(paths, desc="Gathering energy values"):
                    energy_data = torch.load(path)
                    energy_scaler.data.append(energy_data)
        if pitch:
            pitch_scaler = Scaler()
            # Until Python 3.13, pathlib.Path.glob() doesn't work with symlinks: https://github.com/python/cpython/issues/77609
            paths = glob(
                str(self.config.preprocessing.save_dir / "pitch/**/*pitch*"),
                recursive=True,
            )
            if self.cpus > 1:
                logger.info("Gathering pitch values")
                with tqdm_joblib_context(tqdm(desc="Gathering pitch values")):
                    for pitch_data in parallel(
                        delayed(torch.load)(path) for path in paths
                    ):
                        pitch_scaler.data.append(pitch_data)
            else:
                for path in tqdm(paths, desc="Gathering pitch values"):
                    pitch_data = torch.load(path)
                    pitch_scaler.data.append(pitch_data)
        return energy_scaler if energy else energy, pitch_scaler if pitch else pitch

    def normalize_stats(self, energy_scaler: Scaler, pitch_scaler: Scaler):
        """Normalize pitch and energy to unit variance"""
        # Note: this function is IO bound, because it is a tight loop writing small files.
        # Attempts to parallelize it make it much slower, even with only 2 threads.
        logger.info("Scaling dataset statistics...")
        stats = {}
        if energy_scaler:
            energy_stats = energy_scaler.calculate_stats()
            for path in tqdm(
                # Until Python 3.13, pathlib.Path.glob() doesn't work with symlinks: https://github.com/python/cpython/issues/77609
                glob(
                    str(self.config.preprocessing.save_dir / "energy/**/*energy*"),
                    recursive=True,
                ),
                desc="Normalizing energy values",
            ):
                energy = torch.load(path)
                energy = energy_scaler.normalize(energy)
                save_tensor(energy, path)
            stats["energy"] = energy_stats
        if pitch_scaler:
            pitch_stats = pitch_scaler.calculate_stats()
            for path in tqdm(
                # Until Python 3.13, pathlib.Path.glob() doesn't work with symlinks: https://github.com/python/cpython/issues/77609
                glob(
                    str(self.config.preprocessing.save_dir / "pitch/**/*pitch*"),
                    recursive=True,
                ),
                desc="Normalizing pitch values",
            ):
                pitch = torch.load(path)
                pitch = pitch_scaler.normalize(pitch)
                save_tensor(pitch, path)
            stats["pitch"] = pitch_stats

        return stats

    def dataset_sanity_checks(self):
        """Before processing datasets, we should do some sanity checks."""
        for dataset in self.datasets:
            data_dir = Path(dataset.data_dir)
            if not data_dir.exists():
                logger.error(
                    f"Data directory '{data_dir}' does not exist. Please check your config file."
                )
                sys.exit(1)

    def create_path(self, item: dict, folder: str, fn: str) -> Path:
        return (
            self.save_dir
            / folder
            / self.sep.join([item["basename"], item["speaker"], item["language"], fn])
        )

    def process_one_audio(
        self, item: dict, data_dir, sox_effects: list[list]
    ) -> Optional[dict]:
        """Process one audio item

        Return:
           - item if it is found and processed successfully
           - None otherwise, indicating it should be skipped from further processing
        """
        extension = "" if item["basename"].endswith(".wav") else ".wav"
        audio_path = data_dir / (item["basename"] + extension)
        if not audio_path.exists():
            logger.warning(f"File '{item}' is missing and will not be processed.")
            self.counters.increment("missing_files")
            return None

        item = self.get_speaker_and_language(item)
        input_audio_save_path = self.create_path(
            item, "audio", f"audio-{self.input_sampling_rate}.pt"
        )
        output_audio_save_path = self.create_path(
            item, "audio", f"audio-{self.output_sampling_rate}.pt"
        )
        if (
            input_audio_save_path.exists()
            and output_audio_save_path.exists()
            and not self.overwrite
        ):
            audio, sr, seconds = self.load_audio(audio_path)
            self.counters.increment("previously_processed_files")
            self.counters.increment("duration", seconds)
            return item
        if not input_audio_save_path.exists() or self.overwrite:
            input_audio, _ = self.process_audio(
                audio_path,
                resample_rate=self.input_sampling_rate,
                sox_effects=sox_effects,
            )
            if input_audio is None:
                return None
            else:
                save_tensor(input_audio, input_audio_save_path)
        if (
            self.input_sampling_rate != self.output_sampling_rate
            and not output_audio_save_path.exists()
            or self.overwrite
        ):
            output_audio, _ = self.process_audio(
                audio_path,
                resample_rate=self.output_sampling_rate,
                sox_effects=sox_effects,
                update_counters=False,
            )
            if output_audio is not None:
                save_tensor(output_audio, output_audio_save_path)
        return item

    def process_all_audio(self) -> list[dict]:
        """Process all audio across datasets, create a combined, filtered filelist and return it"""
        self.dataset_sanity_checks()
        filtered_filelist: list[dict] = []
        for dataset in tqdm(self.datasets, total=len(self.datasets), desc="Dataset"):
            data_dir = Path(dataset.data_dir)
            filelist = dataset.filelist_loader(dataset.filelist)
            sox_effects = dataset.sox_effects
            if self.debug:
                filelist = filelist[:10]
                logger.info(
                    "Debug flag was set to true, only processing first 10 files"
                )
            logger.info(f"Collecting files for {dataset.label} and processing audio")
            if self.cpus * 3 > len(filelist):
                self.cpus = len(filelist) // 3
            if self.cpus > 1:
                logger.info("Launching parallel processes may take a moment...")
                batch_size = min(100, 1 + len(filelist) // (self.cpus * 2))
                with tqdm_joblib_context(
                    tqdm(
                        desc=f"Processing audio on {self.cpus} CPUs",
                        total=len(filelist),
                    )
                ):
                    processed_items = Parallel(
                        n_jobs=self.cpus,
                        # verbose=10,
                        backend="loky",
                        batch_size=batch_size,
                    )(
                        delayed(self.process_one_audio)(item, data_dir, sox_effects)
                        for item in filelist
                    )
                filtered_filelist.extend(
                    item for item in processed_items if item is not None
                )
            else:
                for item in tqdm(filelist, desc="Processing Audio on 1 CPU"):
                    processed_item = self.process_one_audio(item, data_dir, sox_effects)
                    if processed_item is not None:
                        filtered_filelist.append(processed_item)
        return filtered_filelist

    def process_energy(self, item):
        energy_path = self.create_path(item, "energy", "energy.pt")
        # Always reprocess energy even without self.overwrite, since its results
        # depend on the stats of the whole fileset.
        spec_path = self.create_path(
            item,
            "spec",
            f"spec-{self.input_sampling_rate}-{self.audio_config.spec_type}.pt",
        )
        spec = torch.load(spec_path)
        energy = self.extract_energy(spec)
        if (
            isinstance(self.config, FeaturePredictionConfig)
            and self.config.model.variance_predictors.energy.level == "phone"
            and not self.config.model.learn_alignment
        ):
            dur_path = self.create_path(item, "duration", "duration.pt")
            durs = torch.load(dur_path)
            energy = self.average_data_by_durations(energy, durs)
        save_tensor(energy, energy_path)

    def process_pitch(self, item):
        pitch_path = self.create_path(item, "pitch", "pitch.pt")
        # Always reprocess pitch even without self.overwrite, since its results
        # depend on the stats of the whole fileset.
        audio_path = self.create_path(
            item, "audio", f"audio-{self.input_sampling_rate}.pt"
        )
        audio = torch.load(audio_path)
        pitch = self.extract_pitch(audio)
        if (
            isinstance(self.config, FeaturePredictionConfig)
            and self.config.model.variance_predictors.pitch.level == "phone"
            and not self.config.model.learn_alignment
        ):
            dur_path = self.create_path(item, "duration", "duration.pt")
            durs = torch.load(dur_path)
            pitch = self.average_data_by_durations(pitch, durs)
        save_tensor(pitch, pitch_path)

    def process_attn_prior(self, item):
        attn_prior_path = self.create_path(item, "attn", "attn-prior.pt")
        if attn_prior_path.exists() and not self.overwrite:
            return
        binomial_interpolator = BetaBinomialInterpolator()
        text = self.extract_text_inputs(
            item["text"], self.text_processor, use_pfs=False, quiet=True
        )
        input_spec_path = self.create_path(
            item,
            "spec",
            f"spec-{self.input_sampling_rate}-{self.audio_config.spec_type}.pt",
        )
        input_spec = torch.load(input_spec_path)
        attn_prior = torch.from_numpy(
            binomial_interpolator(input_spec.size(1), text.size(0))
        )
        assert input_spec.size(1) == attn_prior.size(0)
        save_tensor(attn_prior, attn_prior_path)

    def process_text(self, item, use_pfs=False):
        basename = "pfs.pt" if use_pfs else "text.pt"
        text_path = self.create_path(item, "text", basename)
        if text_path.exists() and not self.overwrite:
            return
        text = self.extract_text_inputs(
            item["text"], self.text_processor, use_pfs=use_pfs, quiet=True
        )
        save_tensor(text, text_path)

    def process_spec(self, item):
        input_audio_path = self.create_path(
            item, "audio", f"audio-{self.input_sampling_rate}.pt"
        )
        if not input_audio_path.exists():
            self.counters.increment("skipped_processes")
            logger.info(f"Audio at {input_audio_path} is missing. Skipping...")
            return
        output_audio_path = self.create_path(
            item, "audio", f"audio-{self.output_sampling_rate}.pt"
        )
        input_spec_path = self.create_path(
            item,
            "spec",
            f"spec-{self.input_sampling_rate}-{self.audio_config.spec_type}.pt",
        )

        if input_audio_path != output_audio_path:
            output_spec_path = self.create_path(
                item,
                "spec",
                f"spec-{self.output_sampling_rate}-{self.audio_config.spec_type}.pt",
            )
            if not output_spec_path.exists() or self.overwrite:
                output_audio = torch.load(output_audio_path)
                output_spec = self.extract_spectral_features(
                    output_audio, self.output_spectral_transform
                )
                save_tensor(output_spec, output_spec_path)

        if not input_spec_path.exists() or self.overwrite:
            input_audio = torch.load(input_audio_path)
            input_spec = self.extract_spectral_features(
                input_audio, self.input_spectral_transform
            )
            save_tensor(input_spec, input_spec_path)

    def get_process_fn(self, process):
        if process == "text":
            return functools.partial(self.process_text, use_pfs=False)
        if process == "pfs":
            return functools.partial(self.process_text, use_pfs=True)
        if process == "energy":
            return self.process_energy
        if process == "pitch":
            return self.process_pitch
        if process == "spec":
            return functools.partial(self.process_spec)
        if process == "attn":
            return self.process_attn_prior

    def check_data(self, filelist, word_seg_token=" ", heavy_clip_detction=False):
        data = []
        # speaking rate (words/second, float, scatterplot or bar chart)
        # speaking rate (characters/second, float, scatterplot or bar chart)
        # articulation level (mean energy/speaking rate)
        # unrecognized symbols (bool, list)
        # duration (float, box plot)
        # clipping (float, box plot)
        # silence % (float, box plot)
        for item in tqdm(filelist, desc="Checking Data"):
            data_point = {k: v for k, v in item.items()}
            raw_text = item["text"]
            n_words = len(raw_text.split(word_seg_token))
            n_chars = len(self.text_processor.text_to_sequence(raw_text))
            audio = torch.load(
                self.create_path(item, "audio", f"audio-{self.input_sampling_rate}.pt")
            )
            if heavy_clip_detction:
                _, total_clipping = detect_clipping(audio)
            else:
                # this isn't a great way of detecting clipping,
                # but it's a lot faster than clipdetect
                audio_max = audio.max()
                audio_min = audio.min()
                total_clipping = (
                    audio[audio >= audio_max].size(0)
                    + audio[audio <= audio_min].size(0)
                    - 2
                )
            pitch = torch.load(self.create_path(item, "pitch", "pitch.pt"))
            energy = torch.load(self.create_path(item, "energy", "energy.pt"))
            audio_length_s = len(audio) / self.input_sampling_rate
            data_point["total_clipped_samples"] = total_clipping
            data_point["pitch_min"] = float(pitch.min())
            data_point["pitch_max"] = float(pitch.max())
            data_point["pitch_mean"] = float(pitch.mean())
            data_point["pitch_std"] = float(pitch.std())
            data_point["energy_min"] = float(energy.min())
            data_point["energy_max"] = float(energy.max())
            data_point["energy_mean"] = float(energy.mean())
            data_point["energy_std"] = float(energy.std())
            data_point["duration"] = audio_length_s
            data_point["speaking_rate_word"] = n_words / audio_length_s
            data_point["speaking_rate_char"] = n_chars / audio_length_s
            data_point["articulation_rate_word"] = (
                data_point["energy_mean"] / data_point["speaking_rate_word"]
            )
            data_point["articulation_rate_char"] = (
                data_point["energy_mean"] / data_point["speaking_rate_char"]
            )
            data_point["n_missing_symbols"] = len(
                self.text_processor.get_missing_symbols(raw_text)
            )
            data_point["n_words"] = n_words
            data_point["n_chars"] = n_chars
            data.append(data_point)
        return data

    def load_filelist(self, path: Path):
        try:
            filelist = generic_dict_loader(path)
            if self.debug:
                logger.info(
                    "Debug flag was set to true, only processing first 10 files"
                )
                filelist = filelist[:10]
        except FileNotFoundError:
            logger.error(
                f"A filelist was not found at {path}. "
                "Please try processing your audio again."
            )
            sys.exit(1)
        return filelist

    def preprocess(
        self,
        output_path="filelist.psv",
        cpus=min(5, mp.cpu_count()),
        to_process=list[str],
        overwrite=False,
        debug=False,
    ):
        self.overwrite = overwrite
        self.cpus = cpus
        self.debug = debug
        if not isinstance(output_path, Path):
            output_path = Path(output_path)
        processing_order = ("audio", "text", "pfs", "spec", "attn", "energy", "pitch")
        random.seed(self.config.preprocessing.dataset_split_seed)
        processed_filelist = self.save_dir / output_path.name
        for process in processing_order:
            if process not in to_process:
                continue
            (self.save_dir / process).mkdir(parents=True, exist_ok=True)
            if process == "audio":
                if filelist := self.process_all_audio():
                    write_filelist(filelist, processed_filelist)
                    # sample the validation set and subtract it from the whole dataset to determine the training set
                    random.shuffle(filelist)
                    train_split = int(
                        len(filelist) * self.config.preprocessing.train_split
                    )
                    write_filelist(
                        filelist[:train_split],
                        self.save_dir / f"training_{output_path.name}",
                    )
                    write_filelist(
                        filelist[train_split:],
                        self.save_dir / f"validation_{output_path.name}",
                    )
                    report = self.report()
                    with open(self.save_dir / "summary.txt", "w", encoding="utf8") as f:
                        f.write(report)
                    rich_print(report)
                else:
                    logger.error(
                        "Your filtered audio filelist is empty. Nothing to process."
                    )
                    sys.exit(1)
                # logger.info(f"Audio Filelist len={len(filelist or [])}")
            elif process in ["text", "pfs"]:
                # We split out the "text" step to issue the missing symbol warnings
                filelist = self.load_filelist(processed_filelist)
                process_fn = self.get_process_fn(process)
                missing_symbols_before = Counter(self.text_processor.missing_symbols)
                for f in tqdm(filelist, desc=f"Processing {process} on 1 CPU"):
                    process_fn(f)
                # if only one of "pfs" or "text" is specified, missing_symbols_before
                # will always be empty, but if both are specified this makes sure
                # each process gets only its own missing symbols logged.
                new_missing_symbols = (
                    self.text_processor.missing_symbols - missing_symbols_before
                )
                for symbol, count in new_missing_symbols.items():
                    logger.warning(
                        f"Symbol '{symbol}' occurs {n_times(count)} but was not declared in your configuration so it is being ignored."
                    )
            else:
                # If audio has already been processed, then just read the processed_filelist
                filelist = self.load_filelist(processed_filelist)
                process_fn = self.get_process_fn(process)
                logger.info(f"Processing {process} on {self.cpus} CPUs...")
                # logger.info(f"Filelist len={len(filelist or [])}")
                if self.cpus > 1:
                    batch_size = min(100, 1 + len(filelist) // (self.cpus * 2))
                    with tqdm_joblib_context(
                        tqdm(
                            desc=f"Processing {process} on {self.cpus} CPUs",
                            total=len(filelist),
                        )
                    ):
                        Parallel(
                            n_jobs=self.cpus,
                            backend="loky",
                            batch_size=batch_size,
                        )(delayed(process_fn)(file) for file in filelist)
                else:
                    for f in tqdm(filelist, desc=f"Processing {process} on 1 CPU"):
                        process_fn(f)
        if "audio" in to_process:
            report = f"Here is a report:\n {self.report()}"
            if not self.counters.value("duration"):
                report += "\n\nWARNING: No audio files were processed."
        else:
            report = ""
        rich_print(
            Panel(
                f"You've finished preprocessing: {', '.join(to_process)}. Your files are located at {self.save_dir.absolute()}. {report}",
                title="Congratulations 🎉",
                subtitle="Next Steps Documentation: https://roedoejet.github.io/EveryVoice/guides/custom.html",
                border_style=Style(color="#0B4F19"),
            )
        )
