""" Preprocessor Module that given a filelist containing text, wav, textgrid:
    - extracts log Mel spectral features
    - extracts pitch (phone-level or frame-level)
    - extracts durations
    - extracts energy
    - extracts inputs (ex. phonological feats)
"""
import json
import multiprocessing
import os
from copy import copy
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

# import pyworld as pw
import pandas as pd
import torch  # fix torch imports
from loguru import logger
from tabulate import tabulate
from torch import Tensor, linalg, mean, tensor
from torch.utils.data import DataLoader
from torchaudio import load as load_audio
from torchaudio import save as save_audio
from torchaudio.functional import compute_kaldi_pitch, resample
from torchaudio.sox_effects import apply_effects_tensor
from tqdm import tqdm

from smts.config import ConfigError
from smts.dataloader import FastSpeechDataset, FeaturePredictionDataModule
from smts.text import TextProcessor
from smts.utils import (
    collate_fn,
    dynamic_range_compression_torch,
    get_spectral_transform,
    read_textgrid,
    write_filelist,
)


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.datasets = config["preprocessing"]["source_data"]
        self.save_dir = Path(config["preprocessing"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.missing_files: List[str] = []
        self.skipped_processes = 0
        self.audio_too_short = []
        self.audio_too_long = []
        self.nans = []
        self.duration = 0
        self.audio_config = config["preprocessing"]["audio"]
        self.sep = config["preprocessing"]["value_separator"]
        self.text_processor = TextProcessor(config)
        self.input_sampling_rate = self.audio_config["input_sampling_rate"]
        self.output_sampling_rate = self.audio_config["output_sampling_rate"]
        self.sampling_rate_change = (
            self.output_sampling_rate // self.input_sampling_rate
        )
        # Define Spectral Transform
        # Gah, so many ways to do this: https://github.com/CookiePPP/VocoderComparisons/issues/3

        self.input_spectral_transform = get_spectral_transform(
            self.audio_config["spec_type"],
            self.audio_config["n_fft"],
            self.audio_config["fft_window_frames"],
            self.audio_config["fft_hop_frames"],
            sample_rate=self.input_sampling_rate,
            n_mels=self.audio_config["n_mels"],
            f_min=self.audio_config["f_min"],
            f_max=self.audio_config["f_max"],
        )
        self.output_spectral_transform = get_spectral_transform(
            self.audio_config["spec_type"],
            self.audio_config["n_fft"] * self.sampling_rate_change,
            self.audio_config["fft_window_frames"] * self.sampling_rate_change,
            self.audio_config["fft_hop_frames"] * self.sampling_rate_change,
            sample_rate=self.output_sampling_rate,
            n_mels=self.audio_config["n_mels"],
            f_min=self.audio_config["f_min"],
            f_max=self.audio_config["f_max"],
        )

        if (
            self.input_spectral_transform is None
            or self.output_spectral_transform is None
        ):
            raise ConfigError(
                f"Spectral feature specification {self.audio_config['spec_type']} is not supported. Please edit your config file."
            )

    def load_audio_tensor(self, audio_path: str):
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
        if seconds > self.config["preprocessing"]["audio"]["max_audio_length"]:
            logger.warning(f"Audio too long: {wav_path} ({seconds} seconds)")
            self.audio_too_long.append(os.path.basename(wav_path))
            return None, None
        if seconds < self.config["preprocessing"]["audio"]["min_audio_length"]:
            logger.warning(f"Audio too short: {wav_path} ({seconds} seconds)")
            self.audio_too_long.append(os.path.basename(wav_path))
            return None, None
        self.duration += seconds
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
        if self.config["preprocessing"]["pitch_type"] == "pyworld":
            import pyworld as pw  # This isn't a very good place for an import,

            # but also pyworld is very annoying to install so this is a compromise
            pitch, t = pw.dio(
                audio_tensor.squeeze(0)
                .numpy()
                .astype(
                    np.float64
                ),  # TODO: why are these np.float64, maybe it's just what pw expects?
                self.input_sampling_rate,
                frame_period=self.audio_config["fft_hop_frames"]
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
            pitch = tensor(pitch)
            # TODO: consider interpolating by default when using PyWorld pitch detection
        elif self.config["preprocessing"]["pitch_type"] == "kaldi":
            pitch = compute_kaldi_pitch(
                waveform=audio_tensor,
                sample_rate=self.input_sampling_rate,
                frame_length=self.audio_config["fft_window_frames"]
                / self.input_sampling_rate
                * 1000,
                frame_shift=self.audio_config["fft_hop_frames"]
                / self.input_sampling_rate
                * 1000,
                min_pitch=50,
                max_pitch=400,
            )[0][
                ..., 1
            ]  # TODO: the docs and C Minxhoffer implementation take [..., 0] but this doesn't appear to be the pitch, at least for this version of torchaudio.
        else:
            raise ConfigError(
                f"Sorry, the pitch estimation type '{self.config['preprocessing']['pitch_type']}' is not supported. Please edit your config file."
            )
        return pitch

    def extract_durations(self, textgrid_path: str):
        """Extract durations from a textgrid path
           Don't use tgt package because it ignores silence.

        Args:
            textgrid_path (str): path to a textgrid file
        """
        tg = read_textgrid(textgrid_path)
        phones = tg.get_tier("phones")
        return [
            {
                "start": x[0],
                "end": x[1],
                "dur_ms": (x[1] - x[0]) * 1000,
                "dur_frames": int(
                    (
                        np.round(
                            x[1]
                            * self.input_sampling_rate
                            / self.audio_config["fft_hop_frames"]
                        )
                        - np.round(
                            x[0]
                            * self.input_sampling_rate
                            / self.audio_config["fft_hop_frames"]
                        )
                    )
                ),
                "phone": x[2],
            }
            for x in phones.get_all_intervals()
        ]

    def average_data_by_durations(self, data, durations):
        current_frame_position = 0
        new_data = []
        for duration in durations:
            if duration["dur_frames"] > 0:
                new_data.append(
                    mean(
                        data[
                            current_frame_position : current_frame_position
                            + duration["dur_frames"]
                        ]
                    )
                )
            else:
                new_data.append(1e-7)
            current_frame_position += duration["dur_frames"]
        return tensor(new_data)

    def extract_energy(self, spectral_feature_tensor: Tensor):
        """Given a spectral feature tensor, and durations extract the energy averaged across a phone

        Args:
            spectral_feature_tensor (Tensor): tensor of spectral features extracted from audio
            durations (_type_): _descriptiont    #TODO
        """
        energy = linalg.norm(spectral_feature_tensor, dim=0)
        return energy

    def extract_text_inputs(self, text, use_pfs=False) -> Tensor:
        """Given some text, normalize it, g2p it, and save as one-hot or multi-hot phonological feature vectors

        Args:
            text (str): text
        """
        if use_pfs:
            return torch.Tensor(
                self.text_processor.text_to_phonological_features(text)
            ).long()
        else:
            return torch.Tensor(self.text_processor.text_to_sequence(text)).long()

    def collect_files(self, filelist, data_dir):
        for f in filelist:
            audio_path = data_dir / (f["basename"] + ".wav")
            if not audio_path.exists():
                logger.warning(f"File '{f}' is missing and will not be processed.")
                self.missing_files.append(f)
            else:
                yield f

    def print_duration(self):
        """Convert seconds to a human readable format"""
        seconds = int(self.duration)
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{hours}h {minutes}m {seconds}s"

    def report(self, processed, tablefmt="simple"):
        """Print a report of the dataset processing"""
        headers = ["type", "quantity"]
        table = [
            ["missing files", len(self.missing_files)],
            ["missing symbols", len(self.text_processor.missing_symbols)],
            ["duplicate symbols", len(self.text_processor.duplicate_symbols)],
            ["skipped processes", self.skipped_processes],
            ["nans", self.nans],
            ["audio_too_short", len(self.audio_too_short)],
            ["audio_too_long", len(self.audio_too_long)],
            ["duration", self.print_duration()],
        ]
        return tabulate(table, headers, tablefmt=tablefmt)

    def _preprocess_item(  # noqa: C901
        self,
        item,
        dataset_info,
        process_audio=False,
        process_sox_audio=False,
        process_spec=False,
        process_energy=False,
        process_pitch=False,
        process_duration=False,
        process_pfs=False,
        process_text=False,
        overwrite=False,
    ):
        speaker = "default" if "speaker" not in item else item["speaker"]
        language = "default" if "language" not in item else item["language"]
        item = {**item, **{"speaker": speaker, "language": language}}
        dataset_info["data_dir"] = Path(dataset_info["data_dir"])
        audio = None
        output_audio = None
        spec = None
        input_audio_save_path = self.save_dir / self.sep.join(
            [
                item["basename"],
                speaker,
                language,
                f"audio-{self.input_sampling_rate}.npy",
            ]
        )
        output_audio_save_path = self.save_dir / self.sep.join(
            [
                item["basename"],
                speaker,
                language,
                f"audio-{self.output_sampling_rate}.npy",
            ]
        )
        if process_text:
            save_path = self.save_dir / self.sep.join(
                [item["basename"], speaker, language, "text.npy"]
            )
            if overwrite or not save_path.exists():
                torch.save(
                    self.extract_text_inputs(item["text"]),
                    save_path,
                )
                item["raw_text"] = item["text"]
                item["clean_text"] = self.text_processor.clean_text(item["text"])
            else:
                self.skipped_processes += 1
        if process_pfs:
            save_path = self.save_dir / self.sep.join(
                [item["basename"], speaker, language, "pfs.npy"]
            )
            if overwrite or not save_path.exists():
                torch.save(
                    self.extract_text_inputs(item["text"], use_pfs=True), save_path
                )
            else:
                self.skipped_processes += 1
        if process_sox_audio:
            save_path = self.save_dir / self.sep.join(
                [item["basename"], speaker, language, "processed.wav"]
            )
            if overwrite or not save_path.exists():
                save_audio(
                    save_path,
                    self.process_audio(
                        dataset_info["data_dir"] / (item["basename"] + ".wav"),
                        use_effects=True,
                        sox_effects=dataset_info["sox_effects"],
                    ),
                    self.config["preprocessing"]["audio"]["alignment_sampling_rate"],
                    encoding="PCM_S",
                    bits_per_sample=self.config["preprocessing"]["audio"][
                        "alignment_bit_depth"
                    ],
                )
            else:
                self.skipped_processes += 1
        if process_audio:
            if (
                overwrite
                or not input_audio_save_path.exists()
                or not output_audio_save_path.exists()
            ):
                audio, _ = self.process_audio(
                    dataset_info["data_dir"] / (item["basename"] + ".wav"),
                    resample_rate=self.input_sampling_rate,
                )
                if (
                    audio is None
                ):  # assumes that audio has been handled (ie too long if None)
                    return None
                try:
                    assert torch.any(audio.isnan()).item() is False
                except AssertionError:
                    logger.error(
                        f"Audio for file '{item['basename']}' contains NaNs. Skipping."
                    )
                    self.nans.append(item["basename"])
                    return None
                torch.save(audio, input_audio_save_path)
                if self.input_sampling_rate != self.output_sampling_rate:
                    output_audio, _ = self.process_audio(
                        dataset_info["data_dir"] / (item["basename"] + ".wav"),
                        resample_rate=self.output_sampling_rate,
                    )

                    torch.save(output_audio, output_audio_save_path)
                else:
                    output_audio = audio
            else:
                self.skipped_processes += 1
        if process_spec:
            input_spec_save_path = self.save_dir / self.sep.join(
                [
                    item["basename"],
                    speaker,
                    language,
                    f"spec-{self.input_sampling_rate}-{self.audio_config['spec_type']}.npy",
                ]
            )
            output_spec_save_path = self.save_dir / self.sep.join(
                [
                    item["basename"],
                    speaker,
                    language,
                    f"spec-{self.output_sampling_rate}-{self.audio_config['spec_type']}.npy",
                ]
            )
            if (
                overwrite
                or not input_spec_save_path.exists()
                or not output_spec_save_path.exists()
            ):
                if audio is None:
                    audio, _ = self.process_audio(
                        dataset_info["data_dir"] / (item["basename"] + ".wav"),
                        resample_rate=self.input_sampling_rate,
                    )
                if output_audio is None:
                    output_audio, _ = self.process_audio(
                        dataset_info["data_dir"] / (item["basename"] + ".wav"),
                        resample_rate=self.output_sampling_rate,
                    )
                spec = self.extract_spectral_features(
                    audio, self.input_spectral_transform
                )
                try:
                    assert torch.any(spec.isnan()).item() is False
                except AssertionError:
                    logger.error(
                        f"Spectrogram for file '{item['basename']}' contains NaNs. Skipping."
                    )
                    self.nans.append(item["basename"])
                    return None
                torch.save(
                    spec,
                    input_spec_save_path,
                )
                output_spec = self.extract_spectral_features(
                    output_audio, self.output_spectral_transform
                )
                try:
                    assert torch.any(output_spec.isnan()).item() is False
                except AssertionError:
                    logger.error(
                        f"Spectrogram for file '{item['basename']}' contains NaNs. Skipping."
                    )
                    self.nans.append(item["basename"])
                    return None
                torch.save(output_spec, output_spec_save_path)
            else:
                self.skipped_processes += 1
        if process_pitch:
            save_path = self.save_dir / self.sep.join(
                [item["basename"], speaker, language, "pitch.npy"]
            )
            if overwrite or not save_path.exists():
                if audio is None:
                    audio = self.load_audio_tensor(input_audio_save_path)
                torch.save(
                    self.extract_pitch(audio),
                    save_path,
                )
            else:
                self.skipped_processes += 1
        if process_energy:
            save_path = self.save_dir / self.sep.join(
                [item["basename"], speaker, language, "energy.npy"]
            )
            if overwrite or not save_path.exists():
                if spec is None:
                    try:
                        spec = torch.load(input_spec_save_path)
                    except FileNotFoundError:
                        logger.error(
                            f"Could not find spec file at '{input_spec_save_path}'. Please process the spec first."
                        )
                        exit()
                torch.save(self.extract_energy(spec), save_path),
            else:
                self.skipped_processes += 1
        if process_duration:
            save_path = self.save_dir / self.sep.join(
                [item["basename"], speaker, language, "duration.npy"]
            )
            dur_path = dataset_info["textgrid_dir"] / (item["basename"] + ".TextGrid")
            if not dur_path.exists():
                logger.warning(
                    f"File '{item['basename']}' is missing and will not be processed."
                )
                self.missing_files.append(item)
                return None
            if overwrite or not save_path.exists():
                torch.save(self.extract_durations(dur_path), save_path)
            else:
                self.skipped_processes += 1
        return item

    def compute_priors(self, overwrite=True):
        logger.info("Computing priors...")
        prior_path = self.save_dir / "priors"
        prior_path.mkdir(exist_ok=True)
        filelist = self.config["training"]["feature_prediction"]["filelist_loader"](
            self.config["training"]["feature_prediction"]["filelist"]
        )
        df = pd.DataFrame(filelist)
        ds = FastSpeechDataset(filelist, self.config)
        self._create_priors(df, ds, prior_path)

    def _create_priors(self, data_frame, dataset, prior_path):
        temp_df = copy(data_frame.sort_values("speaker"))
        orig_data = copy(data_frame)
        # check if prior values are already computed
        compute_speakers = []
        for speaker_path in temp_df["speaker"].unique():
            for prior in self.config["model"]["priors"]["prior_types"]:
                if not (prior_path / speaker_path / f"{prior}_prior.npy").exists():
                    compute_speakers.append(speaker_path)
        temp_df = temp_df[temp_df["speaker"].isin(compute_speakers)]

        dl = DataLoader(
            dataset,
            batch_size=10,
            collate_fn=collate_fn,
            num_workers=0,
        )
        speaker_priors = {
            s: {k: [] for k in self.config["model"]["priors"]["prior_types"]}
            for s in temp_df["speaker"].unique()
        }
        pbar = tqdm(
            dl,
            total=len(speaker_priors.keys()),
            maxinterval=1,
            desc="creating prior files (this only runs once)",
        )
        prev_speaker = None
        for item in dl:
            # check if all speakers are the same
            if prev_speaker is None:
                prev_speaker = item["speaker_id"][0]
                prev_path = item["speaker"][0]
            speaker = item["speaker"][0]
            if len(set(item["speaker"])) == 1:
                for prior in self.config["model"]["priors"]["prior_types"]:
                    speaker_priors[speaker][prior] += list(item[f"priors_{prior}"])
            else:
                for i, i_speaker in enumerate(item["speaker"]):
                    for prior in self.config["model"]["priors"]["prior_types"]:
                        speaker_priors[i_speaker][prior].append(
                            item[f"priors_{prior}"][i]
                        )
            if prev_speaker != speaker:
                for prior in self.config["model"]["priors"]["prior_types"]:
                    # save prior to file
                    np.save(
                        prev_path / f"{prior}_prior.npy",
                        speaker_priors[prev_speaker][prior],
                    )
                del speaker_priors[prev_speaker]
                pbar.update(1)
                prev_speaker = speaker
                prev_path = item["speaker_path"][0]
        for speaker in list(speaker_priors.keys()):
            for prior in self.config["model"]["priors"]["prior_types"]:
                np.save(
                    prev_path.parent / speaker / f"{prior}_prior.npy",
                    speaker_priors[speaker][prior],
                )
            del speaker_priors[speaker]
            pbar.update(1)
        pbar.close()
        speaker_priors = {
            s.name: {k: [] for k in self.config["model"]["priors"]["prior_types"]}
            for s in orig_data["speaker"].unique()
        }
        for speaker in tqdm(orig_data["speaker"].unique(), desc="loading priors"):
            for prior in self.config["model"]["priors"]["prior_types"]:
                speaker_priors[speaker.name][prior] = np.load(
                    speaker / f"{prior}_prior.npy"
                )
        return speaker_priors

    def _create_stats_batch(self, x):
        result = {}
        mel_val = x["mel"]
        mel_val[x["silence_mask"]] = np.nan
        result["mel"] = {}
        result["mel"]["mean"] = torch.nanmean(mel_val)
        result["mel"]["std"] = torch.std(mel_val[~torch.isnan(mel_val)])
        for i, var in enumerate(self.config["model"]["variance_adaptor"]["variances"]):
            if (
                self.config["model"]["variance_adaptor"]["variance_transforms"][i]
                == "cwt"
            ):
                var_val = x[f"variances_{var}_original_signal"].float()
            else:
                var_val = x[f"variances_{var}"].float()
            var_val[x["silence_mask"]] = np.nan
            result[var] = {}
            result[var]["min"] = torch.min(var_val[~torch.isnan(var_val)])
            result[var]["max"] = torch.max(var_val[~torch.isnan(var_val)])
            result[var]["mean"] = torch.nanmean(var_val)
            result[var]["std"] = torch.std(var_val[~torch.isnan(var_val)])
            if var in self.config["model"]["priors"]["prior_types"]:
                result[var + "_prior"] = {}
                result[var + "_prior"]["min"] = torch.min(
                    torch.nanmean(var_val, axis=1)
                )
                result[var + "_prior"]["max"] = torch.max(
                    torch.nanmean(var_val, axis=1)
                )
                result[var + "_prior"]["mean"] = torch.mean(
                    torch.nanmean(var_val, axis=1)
                )
                result[var + "_prior"]["std"] = torch.std(
                    torch.nanmean(var_val, axis=1)
                )
        for var in self.config["model"]["priors"]["prior_types"]:
            if var not in self.config["model"]["variance_adaptor"]["variances"]:
                var_val = x[var].float()
                var_val[x["unexpanded_silence_mask"]] = np.nan
                result[var + "_prior"] = {}
                result[var + "_prior"]["min"] = torch.min(
                    torch.nanmean(var_val, axis=1)
                )
                result[var + "_prior"]["max"] = torch.max(
                    torch.nanmean(var_val, axis=1)
                )
                result[var + "_prior"]["mean"] = torch.mean(
                    torch.nanmean(var_val, axis=1)
                )
                result[var + "_prior"]["std"] = torch.std(
                    torch.nanmean(var_val, axis=1)
                )
                del var_val
        return result

    def compute_stats(self, overwrite=True):
        """Compute the mean and standard deviation of the dataset."""
        logger.info("Computing dataset statistics...")
        stat_path = self.save_dir / "stats.json"
        if stat_path.exists() and not overwrite:
            logger.error("Stats already computed. Please re-run with --overwrite=True")
            exit()
        stats = {}
        stat_list = []
        filelist = self.config["training"]["feature_prediction"]["filelist_loader"](
            self.config["training"]["feature_prediction"]["filelist"]
        )
        stats["sample_size"] = len(filelist)
        ds = FastSpeechDataset(filelist, self.config)
        for entry in tqdm(
            DataLoader(
                ds,
                num_workers=0,
                # num_workers=multiprocessing.cpu_count(),
                batch_size=4,
                collate_fn=collate_fn,
                drop_last=True,
            ),
            total=stats["sample_size"] // 4,
            desc="Computing stats",
        ):
            stat_list.append(self._create_stats_batch(entry))
        for key in stat_list[0].keys():
            stats[key] = {}
            for np_stat in stat_list[0][key].keys():
                if np_stat == "std":
                    std_sq = np.array([s[key][np_stat] for s in stat_list]) ** 2
                    stats[key][np_stat] = float(np.sqrt(np.sum(std_sq) / len(std_sq)))
                if np_stat == "mean":
                    stats[key][np_stat] = float(
                        np.mean([s[key]["mean"] for s in stat_list])
                    )
                if np_stat == "min":
                    stats[key][np_stat] = float(
                        np.min([s[key]["min"] for s in stat_list])
                    )
                if np_stat == "max":
                    stats[key][np_stat] = float(
                        np.max([s[key]["max"] for s in stat_list])
                    )
        json.dump(stats, open(stat_path, "w"), indent=4)
        self.stats = stats
        return self.stats

    def preprocess(  # noqa: C901
        self,
        output_path="processed_filelist.psv",
        **kwargs,
    ):
        write_path = self.save_dir / output_path
        if write_path.exists() and not kwargs["overwrite"]:
            logger.error(
                f"Preprocessed filelist at '{write_path}' already exists. Please either set overwrite=True or choose a new path"
            )
            exit()
        # TODO: use multiprocessing
        processed = 0
        files = []
        # Sanity check
        for dataset in self.datasets:
            data_dir = Path(dataset["data_dir"])
            if not data_dir.exists():
                logger.error(
                    f"Data directory '{data_dir}' does not exist. Please check your config file."
                )
                exit()
            # TODO: more sanity checks
        # Actual processing
        for dataset in tqdm(self.datasets, total=len(self.datasets)):
            data_dir = Path(dataset["data_dir"])
            filelist = dataset["filelist_loader"](dataset["filelist"])
            for item in tqdm(
                self.collect_files(filelist, data_dir=data_dir),
                total=len(filelist),
            ):
                item = self._preprocess_item(
                    item,
                    dataset,
                    **kwargs,
                )
                if item is not None:
                    # Add Label
                    if isinstance(dataset["label"], str):
                        item["label"] = dataset["label"]
                    elif callable(dataset["label"]):
                        item["label"] = dataset["label"](item)
                    else:
                        raise ValueError(
                            f"Label for dataset '{dataset['name']}' is neither a string nor a callable."
                        )
                    files.append(item)
                    processed += 1

        logger.info(self.report(processed))
        write_filelist(files, self.save_dir / output_path)
        return files
