import tempfile
from pathlib import Path
from string import ascii_lowercase

from everyvoice.config.shared_types import ContactInformation
from everyvoice.config.text_config import Symbols, TextConfig
from everyvoice.model.e2e.config import FeaturePredictionConfig
from everyvoice.preprocessor import Preprocessor


class PreprocessedInputFixture:
    """
    Preprocess the audio files.
    """

    keep_preprocessed_temp_dir = False
    _preprocess_ran = False

    @classmethod
    def _make_preprocessed_tempdir(cls):
        tempdir_prefix = "tmpdir_PreprocessedInputFixture_"
        if not PreprocessedInputFixture.keep_preprocessed_temp_dir:
            PreprocessedInputFixture.lj_preprocessed_obj = tempfile.TemporaryDirectory(
                prefix=tempdir_prefix, dir="."
            )
            tempdir = PreprocessedInputFixture.lj_preprocessed_obj.name
        else:
            # Alternative tempdir code keeps it after running, for manual inspection:
            tempdir = tempfile.mkdtemp(prefix=tempdir_prefix, dir=".")
            print("tmpdir={}".format(tempdir))
        tempdir = Path(tempdir)
        PreprocessedInputFixture.lj_preprocessed = tempdir / "lj" / "preprocessed"

    @classmethod
    def _prepare_preprocessor(cls):
        PreprocessedInputFixture.data_dir = Path(__file__).parent / "data"
        PreprocessedInputFixture.wavs_dir = (
            PreprocessedInputFixture.data_dir / "lj" / "wavs"
        )
        PreprocessedInputFixture.lj_filelist = (
            PreprocessedInputFixture.lj_preprocessed / "preprocessed_filelist.psv"
        )

        PreprocessedInputFixture.fp_config = FeaturePredictionConfig(
            text=TextConfig(
                symbols=Symbols(
                    ascii_symbols=list(ascii_lowercase),
                    ipa=[
                        "ɔ",
                        "æ",
                        "ɡ",
                        "ɛ",
                        "ð",
                        "ɜ˞",
                        "ʌ",
                        "ɑ",
                        "ɹ",
                        "ʃ",
                        "ɪ",
                        "ʊ",
                        "ʒ",
                    ],
                )
            ),
            contact=ContactInformation(
                contact_name="Test Runner", contact_email="info@everyvoice.ca"
            ),
        )
        PreprocessedInputFixture.fp_config.preprocessing.source_data[0].data_dir = (
            PreprocessedInputFixture.data_dir / "lj" / "wavs"
        )
        PreprocessedInputFixture.fp_config.preprocessing.source_data[0].filelist = (
            PreprocessedInputFixture.data_dir / "metadata.psv"
        )
        PreprocessedInputFixture.fp_config.preprocessing.save_dir = (
            PreprocessedInputFixture.lj_preprocessed
        )

        PreprocessedInputFixture.preprocessor = Preprocessor(
            PreprocessedInputFixture.fp_config
        )

        PreprocessedInputFixture.lj_preprocessed.mkdir(parents=True, exist_ok=True)
        (PreprocessedInputFixture.lj_preprocessed / "duration").symlink_to(
            PreprocessedInputFixture.data_dir / "lj" / "preprocessed" / "duration",
        )

    @classmethod
    def _preprocess(cls):
        """Generate a preprocessed test set that can be used in various test cases."""
        # We only need to actually run this once
        print(
            "================================== GENERATING AUDIO FILES ===================================="
        )
        print(f"====== {PreprocessedInputFixture.lj_filelist = }")

        PreprocessedInputFixture.preprocessor.preprocess(
            output_path=str(PreprocessedInputFixture.lj_filelist),
            cpus=1,
            overwrite=False,
            to_process=("audio", "energy", "pitch", "text", "spec"),
        )

    @classmethod
    def setUpClass(cls):
        if not PreprocessedInputFixture._preprocess_ran:
            PreprocessedInputFixture._make_preprocessed_tempdir()
            PreprocessedInputFixture._prepare_preprocessor()
            PreprocessedInputFixture._preprocess()
            PreprocessedInputFixture._preprocess_ran = True

    @classmethod
    def tearDownClass(cls):
        if not PreprocessedInputFixture.keep_preprocessed_temp_dir:
            PreprocessedInputFixture.lj_preprocessed_obj.cleanup()
