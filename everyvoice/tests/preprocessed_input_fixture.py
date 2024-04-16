import tempfile
from pathlib import Path
from string import ascii_lowercase

from everyvoice.config.preprocessing_config import Dataset, PreprocessingConfig
from everyvoice.config.shared_types import ContactInformation
from everyvoice.config.text_config import Symbols, TextConfig
from everyvoice.model.e2e.config import FeaturePredictionConfig
from everyvoice.preprocessor import Preprocessor


class PreprocessedInputFixture:
    """
    Preprocess the audio files.
    """

    _tempdir = tempfile.TemporaryDirectory(prefix="tmpdir_PreprocessedInputFixture_")
    _preprocess_ran = False
    lj_preprocessed = Path(_tempdir.name)

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
            preprocessing=PreprocessingConfig(
                save_dir=PreprocessedInputFixture.lj_preprocessed,
                source_data=[
                    Dataset(
                        data_dir=PreprocessedInputFixture.wavs_dir,
                        filelist=PreprocessedInputFixture.data_dir / "metadata.psv",
                    )
                ],
            ),
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

        PreprocessedInputFixture.preprocessor = Preprocessor(
            PreprocessedInputFixture.fp_config
        )

    @classmethod
    def _preprocess(cls):
        """Generate a preprocessed test set that can be used in various test cases."""
        # We only need to actually run this once
        if not PreprocessedInputFixture._preprocess_ran:
            PreprocessedInputFixture.preprocessor.preprocess(
                output_path=str(PreprocessedInputFixture.lj_filelist),
                cpus=1,
                overwrite=False,
                to_process=("audio", "energy", "pitch", "text", "spec"),
            )
            PreprocessedInputFixture.lj_preprocessed.mkdir(parents=True, exist_ok=True)
            (PreprocessedInputFixture.lj_preprocessed / "duration").symlink_to(
                PreprocessedInputFixture.data_dir / "lj" / "preprocessed" / "duration",
            )

            PreprocessedInputFixture._preprocess_ran = True

    @classmethod
    def setUpClass(cls):
        PreprocessedInputFixture._prepare_preprocessor()
        PreprocessedInputFixture._preprocess()
