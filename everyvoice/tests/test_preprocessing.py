#!/usr/bin/env python

import tempfile
from pathlib import Path
from unittest import TestCase, main

import torch
from torch import float32

from everyvoice.config.preprocessing_config import (
    AudioConfig,
    AudioSpecTypeEnum,
    PitchCalculationMethod,
    PreprocessingConfig,
)
from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.preprocessor import Preprocessor
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

    # Important side effect: this code must run before all the other tests suites
    # can pass, because it generates the input files for several test cases in this
    # suite and in other test suites.  This works because run_tests.py calls
    # LOAD.loadTestsFromTestCase(PreprocessingTest) regardless of which test
    # suite is being run.
    preprocessor.preprocess(
        output_path=lj_filelist,
        cpus=1,
        overwrite=True,
        to_process=("audio", "energy", "pitch", "text", "spec"),
    )

    def setUp(self) -> None:
        tempdir_prefix = f"tmpdir_{type(self).__name__}_"
        if not self.keep_temp_dir_after_running:
            self.tempdirobj = tempfile.TemporaryDirectory(
                prefix=tempdir_prefix, dir="."
            )
            self.tempdir = self.tempdirobj.name
        else:
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
        self.config = EveryVoiceConfig()
        for entry in self.filelist[1:]:
            # This just applies the SOX effects
            audio, sr = self.preprocessor.process_audio(
                self.wavs_dir / (entry["filename"] + ".wav"),
                use_effects=True,
                sox_effects=self.config.aligner.preprocessing.source_data[
                    0
                ].sox_effects,
            )
            self.assertEqual(sr, 22050)
            self.assertEqual(audio.dtype, float32)

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
        kaldi_config = VocoderConfig(
            preprocessing=PreprocessingConfig(
                pitch_phone_averaging=False, pitch_type=PitchCalculationMethod.kaldi
            )
        )
        pyworld_config = VocoderConfig(
            preprocessing=PreprocessingConfig(
                pitch_phone_averaging=False, pitch_type=PitchCalculationMethod.pyworld
            )
        )
        preprocessor_kaldi = Preprocessor(kaldi_config)
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
            frame_pitch_kaldi = preprocessor_kaldi.extract_pitch(audio.unsqueeze(0))
            kaldi_phone_avg_energy = preprocessor_kaldi.average_data_by_durations(
                frame_pitch_kaldi, durs
            )
            # Ensure same number of frames
            # TODO: Kaldi DOESN'T actually produce the right length tensors here
            # self.assertEqual(
            #     frame_pitch_kaldi.size(0) - 2, feats.size(1)
            # )
            # Ensure avg pitch for each phone
            self.assertEqual(len(durs), kaldi_phone_avg_energy.size(0))
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
        frame_energy_config = VocoderConfig(
            preprocessing=PreprocessingConfig(energy_phone_averaging=False)
        )
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
                # First speaker as one recording
                files = list(tmpdir.glob(f"**/{t}/LJ010/*.pt"))
                self.assertEqual(len(files), 1)
                # Second speaker as 5 recordings
                files = list(tmpdir.glob(f"**/{t}/LJ050/*.pt"))
                self.assertEqual(len(files), 5)


if __name__ == "__main__":
    main()