import tempfile
from pathlib import Path
from unittest import TestCase

from torch import float32

from config.base_config import BaseConfig
from preprocessor import Preprocessor
from utils import read_filelist


class PreprocessingTest(TestCase):
    """Unit tests for preprocessing steps"""

    data_dir = Path(__file__).parent / "data"
    keep_temp_dir_after_running = False

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
        self.filelist = read_filelist(self.data_dir / "metadata.csv")
        self.preprocessor = Preprocessor(BaseConfig())

    def tearDown(self):
        """Clean up the temporary directory"""
        if not self.keep_temp_dir_after_running:
            self.tempdirobj.cleanup()

    def test_read_filelist(self):
        self.assertEqual(self.filelist[0]["filename"], "LJ010-0008")
        self.assertNotIn("speaker", self.filelist[0].keys())

    def test_process_audio_for_alignment(self):
        for entry in self.filelist:
            # This just applies the SOX effects
            audio, sr = self.preprocessor.process_audio(
                self.data_dir / (entry["filename"] + ".wav"), use_effects=True
            )
            self.assertEqual(sr, 16000)
            self.assertEqual(audio.dtype, float32)
            self.assertEqual(audio.size(0), 1)

    def test_process_audio(self):
        for entry in self.filelist:
            audio, sr = self.preprocessor.process_audio(
                self.data_dir / (entry["filename"] + ".wav")
            )
            self.assertEqual(sr, 22050)
            self.assertEqual(audio.dtype, float32)
            self.assertEqual(audio.size(0), 1)

    def test_spectral_feats(self):
        linear_preprocessor = Preprocessor(
            BaseConfig({"preprocessing": {"audio": {"spec_type": "linear"}}})
        )
        complex_preprocessor = Preprocessor(
            BaseConfig({"preprocessing": {"audio": {"spec_type": "raw"}}})
        )

        for entry in self.filelist:
            audio, _ = self.preprocessor.process_audio(
                self.data_dir / (entry["filename"] + ".wav")
            )
            # ming024_feats = np.load(
            #     self.data_dir
            #     / "ming024"
            #     / ("eng-LJSpeech-mel-" + entry["filename"] + ".npy")
            # )
            feats = self.preprocessor.extract_spectral_features(audio)
            linear_feats = linear_preprocessor.extract_spectral_features(audio)
            complex_feats = complex_preprocessor.extract_spectral_features(audio)
            # check data is same number of mels
            self.assertEqual(
                feats.size(0),
                self.preprocessor.config["preprocessing"]["audio"]["n_mels"],
            )
            # Check linear spec has right number of fft bins
            self.assertEqual(
                linear_feats.size(0),
                linear_preprocessor.config["preprocessing"]["audio"]["n_fft"] // 2 + 1,
            )
            # check all same length
            self.assertEqual(feats.size(1), linear_feats.size(1))
            # check all same length
            self.assertEqual(complex_feats.size(1), linear_feats.size(1))

    def test_f0(self):
        preprocessor_kaldi = Preprocessor(
            BaseConfig(
                {"preprocessing": {"f0_phone_averaging": False, "f0_type": "kaldi"}}
            )
        )
        preprocessor_pyworld = Preprocessor(
            BaseConfig(
                {"preprocessing": {"f0_phone_averaging": False, "f0_type": "pyworld"}}
            )
        )

        for entry in self.filelist:
            audio, _ = self.preprocessor.process_audio(
                self.data_dir / (entry["filename"] + ".wav")
            )
            durs = self.preprocessor.extract_durations(
                self.data_dir / (entry["filename"] + ".TextGrid")
            )
            feats = self.preprocessor.extract_spectral_features(audio)
            # ming024_f0 = np.load(
            #     self.data_dir
            #     / "ming024"
            #     / ("eng-LJSpeech-pitch-" + entry["filename"] + ".npy")
            # )
            frame_f0_kaldi = preprocessor_kaldi.extract_f0(audio)
            kaldi_phone_avg_energy = preprocessor_kaldi.average_data_by_durations(
                frame_f0_kaldi, durs
            )
            # Ensure same number of frames
            self.assertEqual(
                frame_f0_kaldi.size(0) - 1, feats.size(1)
            )  # TODO: Why is this -1?
            # Ensure avg f0 for each phone
            self.assertEqual(len(durs), kaldi_phone_avg_energy.size(0))
            frame_f0_pyworld = preprocessor_pyworld.extract_f0(audio)
            pyworld_phone_avg_energy = preprocessor_pyworld.average_data_by_durations(
                frame_f0_pyworld, durs
            )  # TODO: definitely need to interpolate for averaging
            # Ensure avg f0 for each phone
            self.assertEqual(len(durs), pyworld_phone_avg_energy.size(0))
            # Ensure same number of frames
            self.assertEqual(frame_f0_pyworld.size(0), feats.size(1))

    def test_duration(self):
        for entry in self.filelist:
            audio, _ = self.preprocessor.process_audio(
                self.data_dir / (entry["filename"] + ".wav")
            )
            durs = self.preprocessor.extract_durations(
                self.data_dir / (entry["filename"] + ".TextGrid")
            )
            feats = self.preprocessor.extract_spectral_features(audio)
            # ming024_durs = np.load(
            #     self.data_dir
            #     / "ming024"
            #     / ("eng-LJSpeech-duration-" + entry["filename"] + ".npy")
            # )
            # Ensure durations same number of frames as spectral features
            self.assertEqual(feats.size(1), sum(x["dur_frames"] for x in durs))

    def test_energy(self):
        preprocessor = Preprocessor(
            BaseConfig({"preprocessing": {"energy_phone_averaging": False}})
        )
        for entry in self.filelist:
            audio, _ = self.preprocessor.process_audio(
                self.data_dir / (entry["filename"] + ".wav")
            )
            durs = self.preprocessor.extract_durations(
                self.data_dir / (entry["filename"] + ".TextGrid")
            )
            # ming024_energy = np.load(
            #     self.data_dir
            #     / "ming024"
            #     / ("eng-LJSpeech-energy-" + entry["filename"] + ".npy")
            # )
            feats = self.preprocessor.extract_spectral_features(audio)

            frame_energy = preprocessor.extract_energy(feats)
            phone_avg_energy = preprocessor.average_data_by_durations(
                frame_energy.squeeze(), durs
            )
            # Ensure avg energy for each phone
            self.assertEqual(phone_avg_energy.size(0), len(durs))
            # Ensure same number of frames
            self.assertEqual(frame_energy.size(0), feats.size(1))

    def test_sanity(self):
        """TODO: make sanity checking code for each type of data, maybe also data analysis tooling"""
        pass
