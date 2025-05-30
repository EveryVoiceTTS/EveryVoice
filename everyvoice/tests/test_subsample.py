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
        self.assertIn(
            "basename|raw_text|characters|speaker|language|clean_text|label|real_lang",
            result.stdout,
        )
        self.assertIn(
            "LJ050-0269|The essential terms of such memoranda might well be embodied in an Executive order.|The essential terms of such memoranda might well be embodied in an Executive order.|default|default|the essential terms of such memoranda might well be embodied in an executive order.|LJ_TEST|eng",
            result.stdout,
        )
        self.assertIn(
            "LJ050-0270|This Commission can recommend no procedures for the future protection of our Presidents which will guarantee security.|This Commission can recommend no procedures for the future protection of our Presidents which will guarantee security.|default|default|this commission can recommend no procedures for the future protection of our presidents which will guarantee security.|LJ_TEST|eng",
            result.stdout,
        )
        self.assertIn(
            "LJ050-0271|The demands on the President in the execution of His responsibilities in today's world are so varied and complex|The demands on the President in the execution of His responsibilities in today's world are so varied and complex|default|default|the demands on the president in the execution of his responsibilities in today's world are so varied and complex|LJ_TEST|eng",
            result.stdout,
        )
        self.assertIn(
            "LJ050-0272.wav|and the traditions of the office in a democracy such as ours are so deep-seated as to preclude absolute security.|and the traditions of the office in a democracy such as ours are so deep-seated as to preclude absolute security.|default|default|and the traditions of the office in a democracy such as ours are so deep-seated as to preclude absolute security.|LJ_TEST|eng",
            result.stdout,
        )
        self.assertNotIn(
            "LJ050-0273|The Commission has, however, from its examination of the facts of President Kennedy's assassination|The Commission has, however, from its examination of the facts of President Kennedy's assassination|default|default|the commission has, however, from its examination of the facts of president kennedy's assassination|LJ_TEST|eng",
            result.stdout,
        )

    def test_festival(self):
        self.metadata_path = Path(__file__).parent / "data" / "metadata.festival"
        self.wavs_path = Path(__file__).parent / "data" / "lj" / "wavs"
        result = self.runner.invoke(
            app,
            [str(self.metadata_path), str(self.wavs_path), "-d", "7", "-f", "festival"],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            '( LJ050-0269 "The essential terms of such memoranda might well be embodied in an Executive order.")',
            result.stdout,
        )
        self.assertIn(
            '( LJ050-0270 "This Commission can recommend no procedures for the future protection of our Presidents which will guarantee security."  )',
            result.stdout,
        )
        self.assertNotIn(
            '( LJ050-0271 "The demands on the President in the execution of His responsibilities in today\'s world are so varied and complex" )',
            result.stdout,
        )

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

        self.assertIn(
            "basename|raw_text|characters|speaker|language|clean_text|label|real_lang",
            result.stdout,
        )
        self.assertIn(
            "LJ050-0269|The essential terms of such memoranda might well be embodied in an Executive order.|The essential terms of such memoranda might well be embodied in an Executive order.|Speaker_1|default|the essential terms of such memoranda might well be embodied in an executive order.|LJ_TEST|eng",
            result.stdout,
        )
        self.assertIn(
            "LJ050-0272.wav|and the traditions of the office in a democracy such as ours are so deep-seated as to preclude absolute security.|and the traditions of the office in a democracy such as ours are so deep-seated as to preclude absolute security.|Speaker_1|default|and the traditions of the office in a democracy such as ours are so deep-seated as to preclude absolute security.|LJ_TEST|eng",
            result.stdout,
        )
        self.assertNotIn(
            "LJ050-0270|This Commission can recommend no procedures for the future protection of our Presidents which will guarantee security.|This Commission can recommend no procedures for the future protection of our Presidents which will guarantee security.|Speaker_2|default|this commission can recommend no procedures for the future protection Please specify your metadata file's format manuallof our presidents which will guarantee security.|LJ_TEST|eng",
            result.stdout,
        )

    def test_error_validation(self):
        # Test for incorrect file formats
        self.metadata_path = Path(__file__).parent / "data" / "metadata.txt"
        self.wavs_path = Path(__file__).parent / "data" / "lj" / "wavs"
        result = self.runner.invoke(
            app, [str(self.metadata_path), str(self.wavs_path), "-d", "7", "-f", "txt"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertRegex(result.stdout, r"Invalid value for")
        self.assertRegex(
            result.stdout,
            r"txt is not one of psv tsv csv festival".replace(" ", r"[\s\S]*"),
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
