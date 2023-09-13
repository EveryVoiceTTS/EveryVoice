from unittest import TestCase, main

from typer.testing import CliRunner

from everyvoice.cli import app


class CLITest(TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.commands = [
            "new-dataset" "align",
            "synthesize",
            "preprocess",
        ]

    def test_commands_present(self):
        result = self.runner.invoke(app, ["--help"])
        # each command has some help
        for command in self.commands:
            self.assertIn(command, result.stdout)
        # link to docs is present
        self.assertIn("https://docs.everyvoice.ca", result.stdout)


if __name__ == "__main__":
    main()
