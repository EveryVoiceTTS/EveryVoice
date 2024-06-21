import tempfile
from pathlib import Path
from string import ascii_lowercase

from everyvoice.config.preprocessing_config import Dataset, PreprocessingConfig
from everyvoice.config.text_config import Symbols, TextConfig
from everyvoice.model.e2e.config import FeaturePredictionConfig
from everyvoice.preprocessor import Preprocessor
from everyvoice.tests.basic_test_case import BasicTestCase
from everyvoice.utils import collapse_whitespace, lower, nfc_normalize


class PreprocessedAudioFixture:
    """
    A unittest fixture to preprocess the audio files.
    """

    _tempdir = tempfile.TemporaryDirectory(prefix="tmpdir_PreprocessedInputFixture_")
    lj_preprocessed = Path(_tempdir.name)
    _preprocess_ran = False

    data_dir = BasicTestCase.data_dir
    wavs_dir = data_dir / "lj" / "wavs"
    lj_filelist = lj_preprocessed / "preprocessed_filelist.psv"

    fp_config = FeaturePredictionConfig(
        preprocessing=PreprocessingConfig(
            save_dir=lj_preprocessed,
            source_data=[
                Dataset(
                    data_dir=wavs_dir,
                    filelist=data_dir / "metadata.psv",
                    permissions_obtained=True,
                )
            ],
        ),
        text=TextConfig(
            cleaners=[collapse_whitespace, lower, nfc_normalize],
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
            ),
        ),
        contact=BasicTestCase.contact,
    )

    preprocessor = Preprocessor(fp_config)

    @classmethod
    def setUpClass(cls):
        """Generate a preprocessed test set that can be used in various test cases."""
        # We only need to actually run this once
        if not PreprocessedAudioFixture._preprocess_ran:
            PreprocessedAudioFixture.preprocessor.preprocess(
                output_path=str(PreprocessedAudioFixture.lj_filelist),
                cpus=1,
                overwrite=True,
                to_process=("audio", "energy", "pitch", "text", "spec"),
            )
            PreprocessedAudioFixture.lj_preprocessed.mkdir(parents=True, exist_ok=True)
            (PreprocessedAudioFixture.lj_preprocessed / "duration").symlink_to(
                PreprocessedAudioFixture.data_dir / "lj" / "preprocessed" / "duration",
            )

            PreprocessedAudioFixture._preprocess_ran = True
