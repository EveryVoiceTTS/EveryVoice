import json
import tempfile
from pathlib import Path
from unittest import TestCase, main

import jsonschema
from typer.testing import CliRunner

from everyvoice import __file__ as EV_FILE
from everyvoice.cli import SCHEMAS_TO_OUTPUT, app

EV_DIR = Path(EV_FILE).parent


class CLITest(TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.commands = [
            "new-dataset",
            "train",
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

    def test_update_schema(self):
        result = self.runner.invoke(app, ["update-schemas"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("FileExistsError", str(result))
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(app, ["update-schemas", "-o", tmpdir])
            for filename, obj in SCHEMAS_TO_OUTPUT.items():
                with open(Path(tmpdir) / filename, encoding="utf8") as f:
                    schema = json.load(f)
                # serialize the model to json and then validate against the schema
                self.assertIsNone(
                    jsonschema.validate(
                        json.loads(obj().model_dump_json()), schema=schema
                    )
                )


if __name__ == "__main__":
    main()
