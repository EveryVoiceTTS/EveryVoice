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
    _delme = tempfile.TemporaryDirectory()

    @classmethod
    def _make_preprocessed_tempdir(cls):
        tempdir_prefix = f"tmpdir_{type(cls).__name__}_"
        if not cls.keep_preprocessed_temp_dir:
            cls.lj_preprocessed_obj = tempfile.TemporaryDirectory(
                prefix=tempdir_prefix, dir="."
            )
            tempdir = cls.lj_preprocessed_obj.name
        else:
            # Alternative tempdir code keeps it after running, for manual inspection:
            tempdir = tempfile.mkdtemp(prefix=tempdir_prefix, dir=".")
            print("tmpdir={}".format(tempdir))
        tempdir = Path(tempdir)
        cls.lj_preprocessed = tempdir / "lj" / "preprocessed"

    @classmethod
    def _prepare_preprocessor(cls):
        cls.data_dir = Path(__file__).parent / "data"
        cls.wavs_dir = cls.data_dir / "lj" / "wavs"
        cls.lj_filelist = cls.lj_preprocessed / "preprocessed_filelist.psv"

        cls.fp_config = FeaturePredictionConfig(
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
        cls.fp_config.preprocessing.source_data[0].data_dir = (
            cls.data_dir / "lj" / "wavs"
        )
        cls.fp_config.preprocessing.source_data[0].filelist = (
            cls.data_dir / "metadata.psv"
        )
        cls.fp_config.preprocessing.save_dir = cls.lj_preprocessed

        cls.preprocessor = Preprocessor(cls.fp_config)

        cls.lj_preprocessed.mkdir(parents=True, exist_ok=True)
        (cls.lj_preprocessed / "duration").symlink_to(
            cls.data_dir / "lj" / "preprocessed" / "duration",
        )

    @classmethod
    def _preprocess(cls):
        """Generate a preprocessed test set that can be used in various test cases."""
        # We only need to actually run this once
        if not cls._preprocess_ran:
            print(
                "================================== GENERATING AUDIO FILES ===================================="
            )
            print(
                f"=============== _preprocess_ran: {id(cls._preprocess_ran)} {cls._preprocess_ran}"
            )
            print(cls.lj_filelist)

            cls.preprocessor.preprocess(
                output_path=str(cls.lj_filelist),
                cpus=1,
                overwrite=False,
                to_process=("audio", "energy", "pitch", "text", "spec"),
            )
            cls._preprocess_ran = True

    @classmethod
    def setUpClass(cls):
        print("PreprocessedInputFixture:", cls._delme)
        cls._make_preprocessed_tempdir()
        cls._prepare_preprocessor()
        cls._preprocess()

    @classmethod
    def tearDownClass(cls):
        if not cls.keep_preprocessed_temp_dir:
            cls.lj_preprocessed_obj.cleanup()
