#!/usr/bin/env python

import tempfile
from pathlib import Path
from unittest import TestCase, main

import torch
from pydantic_core._pydantic_core import ValidationError
from torch import float32

from everyvoice.config.preprocessing_config import (
    AudioConfig,
    AudioSpecTypeEnum,
    PreprocessingConfig,
)
from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.preprocessor import Preprocessor
from everyvoice.tests.stubs import capture_stdout, mute_logger
from everyvoice.utils import read_filelist


class PreprocessingTest(TestCase):
    """Unit tests for preprocessing steps"""

    data_dir = Path(__file__).parent / "data"
    wavs_dir = data_dir / "lj" / "wavs"
    lj_preprocessed = data_dir / "lj" / "preprocessed"
    lj_filelist = lj_preprocessed / "preprocessed_filelist.psv"
    keep_temp_dir_after_running = False

    fp_config = EveryVoiceConfig().feature_prediction
    fp_config.preprocessing.source_data[0].data_dir = data_dir / "lj" / "wavs"
    fp_config.preprocessing.source_data[0].filelist = data_dir / "metadata.csv"
    fp_config.preprocessing.save_dir = lj_preprocessed
    preprocessor = Preprocessor(fp_config)

    # Important side effect: this code must run before test_dataloader.py
    # can pass, because it generates the input files for several test cases in this
    # suite and in that test suite.
    preprocessor.preprocess(
        output_path=lj_filelist,
        cpus=1,
        overwrite=False,
        to_process=("audio", "energy", "pitch", "text", "spec"),
    )

    def setUp(self) -> None:
        tempdir_prefix = f"tmpdir_{type(self).__name__}_"
        if not self.keep_temp_dir_after_running:
            self.tempdirobj = tempfile.TemporaryDirectory(
                prefix=tempdir_prefix, dir="."
            )
            self.tempdir = self.tempdirobj.name
        else:  # pragma: no cover
            # Alternative tempdir code keeps it after running, for manual inspection:
            self.tempdir = tempfile.mkdtemp(prefix=tempdir_prefix, dir=".")
            print(f"tmpdir={self.tempdir}")
        self.tempdir = Path(self.tempdir)  # type: ignore
        self.filelist = read_filelist(self.data_dir / "metadata.csv")  # type: ignore

    def tearDown(self):
        """Clean up the temporary directory"""
        if not self.keep_temp_dir_after_running:
            self.tempdirobj.cleanup()

    # def test_compute_stats(self):
    #     feat_prediction_config = EveryVoiceConfig.load_config_from_path().feature_prediction
    #     preprocessor = Preprocessor(feat_prediction_config)
    #     preprocessor.compute_stats()
    # self.assertEqual(
    #     self.preprocessor.config["preprocessing"]["audio"]["mel_mean"],
    #     -4.018,
    #     places=3,
    # )
    # self.assertEqual(
    #     self.preprocessor.config["preprocessing"]["audio"]["mel_std"],
    #     4.017,
    #     places=3,
    # )

    def test_read_filelist(self):
        self.assertEqual(self.filelist[1]["filename"], "LJ050-0269")
        self.assertNotIn("speaker", self.filelist[0].keys())

    def test_process_audio_for_alignment(self):
        config = EveryVoiceConfig()
        for entry in self.filelist[1:]:
            # This just applies the SOX effects
            audio, sr = self.preprocessor.process_audio(
                self.wavs_dir / (entry["filename"] + ".wav"),
                use_effects=True,
                sox_effects=config.aligner.preprocessing.source_data[0].sox_effects,
            )
            self.assertEqual(sr, 22050)
            self.assertEqual(audio.dtype, float32)

    def test_process_empty_audio(self):
        for fn in ["empty.wav", "zeros.wav"]:
            audio, sr = self.preprocessor.process_audio(self.data_dir / fn)
            self.assertEqual(audio, None)
            self.assertEqual(sr, None)

    def test_process_audio(self):
        for entry in self.filelist[1:]:
            audio, sr = self.preprocessor.process_audio(
                self.wavs_dir / (entry["filename"] + ".wav")
            )
            self.assertEqual(sr, 22050)
            self.assertEqual(audio.dtype, float32)

    def test_spectral_feats(self):
        linear_vocoder_config = VocoderConfig(
            preprocessing=PreprocessingConfig(
                audio=AudioConfig(spec_type=AudioSpecTypeEnum.linear)
            )
        )
        complex_vocoder_config = VocoderConfig(
            preprocessing=PreprocessingConfig(
                audio=AudioConfig(spec_type=AudioSpecTypeEnum.raw)
            )
        )
        linear_preprocessor = Preprocessor(linear_vocoder_config)
        complex_preprocessor = Preprocessor(complex_vocoder_config)

        for entry in self.filelist[1:]:
            audio, _ = self.preprocessor.process_audio(
                self.wavs_dir / (entry["filename"] + ".wav")
            )
            # ming024_feats = np.load(
            #     self.data_dir
            #     / "ming024"
            #     / ("eng-LJSpeech-mel-" + entry["filename"] + ".npy")
            # )
            feats = self.preprocessor.extract_spectral_features(
                audio, self.preprocessor.input_spectral_transform
            )
            linear_feats = linear_preprocessor.extract_spectral_features(
                audio, linear_preprocessor.input_spectral_transform
            )
            complex_feats = complex_preprocessor.extract_spectral_features(
                audio, complex_preprocessor.input_spectral_transform, normalize=False
            )
            # check data is same number of mels
            self.assertEqual(
                feats.size(0),
                self.preprocessor.config.preprocessing.audio.n_mels,
            )
            # Check linear spec has right number of fft bins
            self.assertEqual(
                linear_feats.size(0),
                linear_preprocessor.config.preprocessing.audio.n_fft // 2 + 1,
            )
            # check all same length
            self.assertEqual(feats.size(1), linear_feats.size(1))
            # check all same length
            self.assertEqual(complex_feats.size(1), linear_feats.size(1))

    def test_pitch(self):
        pyworld_config = VocoderConfig(preprocessing=PreprocessingConfig())
        preprocessor_pyworld = Preprocessor(pyworld_config)

        for entry in self.filelist[1:]:
            audio, _ = self.preprocessor.process_audio(
                self.wavs_dir / (entry["filename"] + ".wav")
            )
            dur_path = (
                self.lj_preprocessed
                / "duration"
                / self.preprocessor.sep.join(
                    [
                        entry["filename"],
                        entry.get("speaker", "default"),
                        entry.get("language", "default"),
                        "duration.pt",
                    ]
                )
            )
            durs = torch.load(dur_path)
            feats = self.preprocessor.extract_spectral_features(
                audio, self.preprocessor.input_spectral_transform
            )
            # ming024_pitch = np.load(
            #     self.data_dir
            #     / "ming024"
            #     / ("eng-LJSpeech-pitch-" + entry["filename"] + ".npy")
            # )
            # Ensure avg pitch for each phone
            frame_pitch_pyworld = preprocessor_pyworld.extract_pitch(audio)
            pyworld_phone_avg_energy = preprocessor_pyworld.average_data_by_durations(
                frame_pitch_pyworld, durs
            )
            # Ensure avg pitch for each phone
            self.assertEqual(len(durs), pyworld_phone_avg_energy.size(0))
            # Ensure same number of frames
            self.assertEqual(frame_pitch_pyworld.size(0), feats.size(1))

    # TODO: test nans: torch.any(torch.Tensor([[torch.nan, 2]]).isnan())

    def test_duration(self):
        for entry in self.filelist[1:]:
            audio, _ = self.preprocessor.process_audio(
                self.wavs_dir / (entry["filename"] + ".wav")
            )
            dur_path = (
                self.lj_preprocessed
                / "duration"
                / self.preprocessor.sep.join(
                    [
                        entry["filename"],
                        entry.get("speaker", "default"),
                        entry.get("language", "default"),
                        "duration.pt",
                    ]
                )
            )
            durs = torch.load(dur_path)
            feats = self.preprocessor.extract_spectral_features(
                audio, self.preprocessor.input_spectral_transform
            )
            # ming024_durs = np.load(
            #     self.data_dir
            #     / "ming024"
            #     / ("eng-LJSpeech-duration-" + entry["filename"] + ".npy")
            # )
            # Ensure durations same number of frames as spectral features
            # note: this is off by a few frames due to mismatches in hop size between the aligner the test data
            # was trained with and the settings defined by the spectral transform function here.
            # It would be a problem if it weren't  but it's not really relevant since we're using jointly learned alignments now.
            self.assertTrue(feats.size(1) - int(sum(durs)) <= 10)

    def test_energy(self):
        frame_energy_config = VocoderConfig(preprocessing=PreprocessingConfig())
        preprocessor = Preprocessor(frame_energy_config)
        for entry in self.filelist[1:]:
            audio, _ = self.preprocessor.process_audio(
                self.wavs_dir / (entry["filename"] + ".wav")
            )
            dur_path = (
                self.lj_preprocessed
                / "duration"
                / self.preprocessor.sep.join(
                    [
                        entry["filename"],
                        entry.get("speaker", "default"),
                        entry.get("language", "default"),
                        "duration.pt",
                    ]
                )
            )
            durs = torch.load(dur_path)
            # ming024_energy = np.load(
            #     self.data_dir
            #     / "ming024"
            #     / ("eng-LJSpeech-energy-" + entry["filename"] + ".npy")
            # )
            feats = self.preprocessor.extract_spectral_features(
                audio, self.preprocessor.input_spectral_transform
            )

            frame_energy = preprocessor.extract_energy(feats)
            phone_avg_energy = preprocessor.average_data_by_durations(
                frame_energy, durs
            )
            # Ensure avg energy for each phone
            self.assertEqual(phone_avg_energy.size(0), len(durs))
            # Ensure same number of frames
            self.assertEqual(frame_energy.size(0), feats.size(1))

    def test_sanity(self):
        """TODO: make sanity checking code for each type of data, maybe also data analysis tooling"""
        pass

    def test_incremental_preprocess(self):
        with tempfile.TemporaryDirectory(prefix="incremental", dir=".") as tmpdir:
            tmpdir = Path(tmpdir)
            lj_preprocessed = tmpdir / "preprocessed"
            lj_filelist = lj_preprocessed / "preprocessed_filelist.psv"

            fp_config = EveryVoiceConfig().feature_prediction
            fp_config.preprocessing.source_data[0].data_dir = (
                self.data_dir / "lj" / "wavs"
            )
            full_filelist = self.data_dir / "metadata.csv"
            partial_filelist = tmpdir / "partial-metadata.psv"
            with open(partial_filelist, mode="w") as f_out:
                with open(full_filelist) as f_in:
                    lines = list(f_in)
                    for line in lines[:4]:
                        f_out.write(line)
            fp_config.preprocessing.source_data[0].filelist = partial_filelist
            fp_config.preprocessing.save_dir = lj_preprocessed

            to_process = ("audio", "energy", "pitch", "attn", "text", "spec")
            with capture_stdout() as output, mute_logger("everyvoice.preprocessor"):
                Preprocessor(fp_config).preprocess(
                    output_path=lj_filelist, cpus=1, to_process=to_process
                )
            self.assertRegex(output.getvalue(), r"processed files *3")
            self.assertRegex(output.getvalue(), r"previously processed files *0")

            fp_config.preprocessing.source_data[0].filelist = full_filelist
            with capture_stdout() as output, mute_logger("everyvoice.preprocessor"):
                Preprocessor(fp_config).preprocess(
                    output_path=lj_filelist, cpus=1, to_process=to_process
                )
            self.assertRegex(output.getvalue(), r"processed files *2")
            self.assertRegex(output.getvalue(), r"previously processed files *3")
            with capture_stdout() as output, mute_logger("everyvoice.preprocessor"):
                Preprocessor(fp_config).preprocess(
                    output_path=lj_filelist,
                    cpus=1,
                    overwrite=True,
                    to_process=to_process,
                    debug=True,
                )
            self.assertRegex(output.getvalue(), r"processed files *5")
            self.assertRegex(output.getvalue(), r"previously processed files *0")

    def test_gotta_do_audio_first(self):
        with tempfile.TemporaryDirectory(prefix="missing_audio", dir=".") as tmpdir:
            tmpdir = Path(tmpdir)
            preprocessed = tmpdir / "preprocessed"
            filelist = preprocessed / "preprocessed_filelist.psv"

            fp_config = EveryVoiceConfig().feature_prediction
            fp_config.preprocessing.source_data[0].data_dir = (
                self.data_dir / "lj" / "wavs"
            )
            full_filelist = self.data_dir / "metadata.csv"
            fp_config.preprocessing.source_data[0].filelist = full_filelist
            fp_config.preprocessing.save_dir = preprocessed

            to_process_no_audio = ("energy", "pitch", "attn", "text", "spec")
            with self.assertRaises(SystemExit), capture_stdout():
                Preprocessor(fp_config).preprocess(
                    output_path=filelist, cpus=1, to_process=to_process_no_audio
                )

    def test_empty_preprocess(self):
        # Test case where the file list is not empty but after filtering
        # silence, the result is empty. The behaviour of the code base is not
        # super satisfying, we exit when we try to read
        # preprocessed/filelist.psv and it's not there, rather than catching the
        # fact that we're trying to write an empty list.
        with tempfile.TemporaryDirectory(prefix="empty", dir=".") as tmpdir:
            tmpdir = Path(tmpdir)
            preprocessed = tmpdir / "preprocessed"
            filelist = preprocessed / "preprocessed_filelist.psv"

            fp_config = EveryVoiceConfig().feature_prediction
            fp_config.preprocessing.source_data[0].data_dir = self.data_dir
            input_filelist = tmpdir / "empty-metadata.psv"
            with open(input_filelist, mode="w") as f:
                print("basename|raw_text|text|speaker|language", file=f)
                print("empty|foo bar baz|foo bar baz|noone|und", file=f)
            fp_config.preprocessing.source_data[0].filelist = input_filelist
            fp_config.preprocessing.save_dir = preprocessed

            to_process = ("audio", "energy", "pitch", "attn", "text", "spec")
            with self.assertRaises(SystemExit), capture_stdout():
                Preprocessor(fp_config).preprocess(
                    output_path=filelist, cpus=1, to_process=to_process
                )

    def test_train_split(self):
        """
        PreprocessingConfig's train_split should be [0., 1.].
        """
        config = PreprocessingConfig(train_split=0.5)
        self.assertEqual(config.train_split, 0.5)

        config = PreprocessingConfig(train_split=0.0)
        self.assertEqual(config.train_split, 0.0)

        config = PreprocessingConfig(train_split=1.0)
        self.assertEqual(config.train_split, 1.0)

        with self.assertRaises(ValidationError), capture_stdout() as cout:
            config = PreprocessingConfig(train_split=-0.1)
            self.assertIn("Input should be greater than or equal to 0", cout.getvalue())
        with self.assertRaises(ValidationError), capture_stdout() as cout:
            config = PreprocessingConfig(train_split=1.1)
            self.assertIn("Input should be less than or equal to 1", cout.getvalue())


class PreprocessingHierarchyTest(TestCase):
    def test_hierarchy(self):
        """Unit tests for preprocessing steps"""

        with tempfile.TemporaryDirectory(prefix="hierarchy", dir=".") as tmpdir:
            tmpdir = Path(tmpdir)
            data_dir = Path(__file__).parent / "data"
            wavs_dir = data_dir / "hierarchy" / "wavs"
            preprocessed_dir = tmpdir / "hierarchy" / "preprocessed"
            filelist = preprocessed_dir / "preprocessed_filelist.psv"

            fp_config = EveryVoiceConfig().feature_prediction
            fp_config.preprocessing.source_data[0].data_dir = wavs_dir
            fp_config.preprocessing.source_data[0].filelist = (
                data_dir / "hierarchy" / "metadata.csv"
            )
            fp_config.preprocessing.save_dir = preprocessed_dir
            preprocessor = Preprocessor(fp_config)

            with mute_logger("everyvoice.preprocessor"), capture_stdout():
                preprocessor.preprocess(
                    output_path=filelist,
                    cpus=2,
                    overwrite=True,
                    # to_process=("audio", "energy", "pitch", "text", "spec"),
                    # to_process=("audio", "text", "pfs", "spec", "attn", "energy", "pitch"),
                    to_process=("audio", "text", "spec", "attn", "energy", "pitch"),
                )

            for t in ("audio", "text", "spec", "attn", "energy", "pitch"):
                # There are two speakers
                sources = [d.name for d in tmpdir.glob(f"**/{t}/*")]
                self.assertSetEqual(set(sources), set(("LJ010", "LJ050")))
                # First speaker has one recording
                files = list(tmpdir.glob(f"**/{t}/LJ010/*.pt"))
                self.assertEqual(len(files), 1)
                # Second speaker has 5 recordings
                files = list(tmpdir.glob(f"**/{t}/LJ050/*.pt"))
                self.assertEqual(len(files), 5)


if __name__ == "__main__":
    main()
