from pathlib import Path
from unittest import TestCase

from typer.testing import CliRunner

from everyvoice.tests.regression.subsample import app


class SubsampleTest(TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    # In this test case we use .psv, but we assume this implies .tsv and .csv will work as well.
    def test_sv(self):
        self.metadata_path = Path(__file__).parent / "data" / "metadata.psv"
        self.wavs_path = Path(__file__).parent / "data" / "lj" / "wavs"
        result = self.runner.invoke(
            app,
            [
                str(self.metadata_path),
                str(self.wavs_path),
                "--header",
                "-d",
                "25",
                "-f",
                "psv",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("basename|", result.stdout)
        self.assertIn("LJ050-0269|", result.stdout)
        self.assertIn("LJ050-0270|", result.stdout)
        self.assertIn("LJ050-0271|", result.stdout)
        self.assertIn("LJ050-0272.wav|", result.stdout)
        self.assertNotIn("LJ050-0273|", result.stdout)

    def test_festival(self):
        self.metadata_path = Path(__file__).parent / "data" / "metadata.festival"
        self.wavs_path = Path(__file__).parent / "data" / "lj" / "wavs"
        result = self.runner.invoke(
            app,
            [str(self.metadata_path), str(self.wavs_path), "-d", "7", "-f", "festival"],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("LJ050-0269", result.stdout)
        self.assertIn("LJ050-0270", result.stdout)
        self.assertNotIn("LJ050-0271", result.stdout)

    def test_help(self):
        result = self.runner.invoke(app, ["--help"])
        self.assertIn("Standalone test script for subsampling corpora", result.stdout)

    def test_speakerid(self):
        self.metadata_path = (
            Path(__file__).parent / "data" / "metadata_different_speakers.psv"
        )
        self.wavs_path = Path(__file__).parent / "data" / "lj" / "wavs"
        result = self.runner.invoke(
            app,
            [
                str(self.metadata_path),
                str(self.wavs_path),
                "--header",
                "-d",
                "6",
                "-f",
                "psv",
                "-s",
                "3",
                "-i",
                "Speaker_1",
            ],
        )

        self.assertIn("basename|", result.stdout)
        self.assertIn("LJ050-0269|", result.stdout)
        self.assertIn("LJ050-0272.wav|", result.stdout)
        self.assertNotIn("LJ050-0270|", result.stdout)

    def test_error_validation(self):
        # Test for incorrect file formats
        self.metadata_path = Path(__file__).parent / "data" / "metadata.txt"
        self.wavs_path = Path(__file__).parent / "data" / "lj" / "wavs"
        result = self.runner.invoke(
            app, [str(self.metadata_path), str(self.wavs_path), "-d", "7", "-f", "txt"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid value for", result.stdout)
        self.assertRegex(
            result.stdout,
            r"(?s)txt is not one of psv tsv csv festival".replace(" ", r".*"),
        )

        # Festival format with speaker id is incompatible
        self.metadata_path = Path(__file__).parent / "data" / "metadata.festival"
        self.wavs_path = Path(__file__).parent / "data" / "lj" / "wavs"
        result = self.runner.invoke(
            app,
            [
                str(self.metadata_path),
                str(self.wavs_path),
                "-d",
                "7",
                "-f",
                "festival",
                "-s",
                "3",
                "--speakerid",
                "Speaker_1",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertRegex(
            result.stdout,
            r"Invalid value: Festival formatted files cannot have a speaker id.".replace(
                " ", r"[\s\S]*"
            ),
        )

        # A Wav File cannot be found
        self.metadata_path = Path(__file__).parent / "data" / "metadata.psv"
        self.wavs_path = Path(__file__).parent / "data" / "lj" / "wavs"
        result = self.runner.invoke(
            app,
            [
                str(self.metadata_path),
                str(self.wavs_path),
                "-d",
                "7",
                "-f",
                "psv",
            ],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertRegex(
            result.stdout,
            r"A \.wav file could not be found".replace(" ", r"[\s\S]*"),
        )
