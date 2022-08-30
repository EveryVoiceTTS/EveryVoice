import tempfile
from pathlib import Path
from unittest import TestCase

from config.base_config import BaseConfig
from preprocessor import Preprocessor
from torch import float32
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
            # This just applies the SOX effects, which currently resample to 16000
            audio, sr = self.preprocessor.process_audio_for_alignment(
                self.data_dir / (entry["filename"] + ".wav")
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

    def test_feats(self):
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
            feats = self.preprocessor.extract_spectral_features(audio)
            linear_feats = linear_preprocessor.extract_spectral_features(audio)
            complex_feats = complex_preprocessor.extract_spectral_features(audio)
            self.assertEqual(feats.size(0), 1)
            self.assertEqual(
                feats.size(1),
                self.preprocessor.config["preprocessing"]["audio"]["n_mels"],
            )  # check data is same number of mels
            self.assertEqual(
                linear_feats.size(1),
                linear_preprocessor.config["preprocessing"]["audio"]["n_fft"] // 2 + 1,
            )
            self.assertEqual(
                feats.size(2), linear_feats.size(2)
            )  # check all same length
            self.assertEqual(
                complex_feats.size(2), linear_feats.size(2)
            )  # check all same length

    def test_f0(self):
        pass

    def test_duration(self):
        pass

    def test_energy(self):
        frame_preprocessor = Preprocessor(
            BaseConfig({"preprocessing": {"energy_type": "frame"}})
        )
        for entry in self.filelist:
            audio, _ = self.preprocessor.process_audio(
                self.data_dir / (entry["filename"] + ".wav")
            )
            feats = self.preprocessor.extract_spectral_features(audio)
            # phone_energy = self.preprocessor.extract_energy(
            #     feats, []
            # )  # TODO: need durations
            frame_energy = frame_preprocessor.extract_energy(feats, None)
            self.assertEqual(frame_energy.size(1), feats.size(2))

    def test_text(self):
        pass
