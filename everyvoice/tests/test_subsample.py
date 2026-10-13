#!/usr/bin/env python

import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from everyvoice.tests.regression.subsample import app


class TestSubsample:
    def setup_method(self) -> None:
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
        assert result.exit_code == 0
        assert "basename|" in result.stdout
        assert "LJ050-0269|" in result.stdout
        assert "LJ050-0270|" in result.stdout
        assert "LJ050-0271|" in result.stdout
        assert "LJ050-0272.wav|" in result.stdout
        assert "LJ050-0273|" not in result.stdout

    def test_festival(self):
        self.metadata_path = Path(__file__).parent / "data" / "metadata.festival"
        self.wavs_path = Path(__file__).parent / "data" / "lj" / "wavs"
        result = self.runner.invoke(
            app,
            [str(self.metadata_path), str(self.wavs_path), "-d", "7", "-f", "festival"],
        )

        assert result.exit_code == 0
        assert "LJ050-0269" in result.stdout
        assert "LJ050-0270" in result.stdout
        assert "LJ050-0271" not in result.stdout

    def test_help(self):
        result = self.runner.invoke(app, ["--help"])
        assert "Standalone test script for subsampling corpora" in result.stdout

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

        assert "basename|" in result.stdout
        assert "LJ050-0269|" in result.stdout
        assert "LJ050-0272.wav|" in result.stdout
        assert "LJ050-0270|" not in result.stdout

    def test_error_validation(self):
        # Test for incorrect file formats
        self.metadata_path = Path(__file__).parent / "data" / "metadata.txt"
        self.wavs_path = Path(__file__).parent / "data" / "lj" / "wavs"
        result = self.runner.invoke(
            app, [str(self.metadata_path), str(self.wavs_path), "-d", "7", "-f", "txt"]
        )
        assert result.exit_code != 0
        assert "Invalid value for" in result.output
        assert re.search(
            r"(?s)txt is not one of psv tsv csv festival".replace(" ", r".*"),
            result.output,
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

        assert result.exit_code != 0
        assert re.search(
            r"Invalid value: Festival formatted files cannot have a speaker id.".replace(
                " ", r"[\s\S]*"
            ),
            result.output,
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
        assert result.exit_code != 0
        assert re.search(
            r"A \.wav file could not be found".replace(" ", r"[\s\S]*"),
            result.output,
        )


if __name__ == "__main__":
    pytest.main(sys.argv)
