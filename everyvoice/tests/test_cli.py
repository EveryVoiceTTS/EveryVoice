import json
import os
import subprocess
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import jsonschema
import yaml
from pydantic import ValidationError
from pytorch_lightning import Trainer
from typer.testing import CliRunner
from yaml import CLoader as Loader

# required for `./run_tests.py cli` to work, otherwise test_inspect_checkpoint
# fails with an Intel MKL FATAL ERROR saying it cannot load libtorch_cpu.so
import everyvoice.tests.test_model  # noqa
from everyvoice import __file__ as EV_FILE
from everyvoice.base_cli.helpers import save_configuration_to_log_dir
from everyvoice.cli import SCHEMAS_TO_OUTPUT, app
from everyvoice.config.shared_types import ContactInformation
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.config import (
    FastSpeech2Config,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
    FastSpeech2,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions_heavy import (
    Stats,
    StatsInfo,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import HiFiGANConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import HiFiGAN
from everyvoice.tests.stubs import capture_logs, mute_logger
from everyvoice.wizard import (
    SPEC_TO_WAV_CONFIG_FILENAME_PREFIX,
    TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX,
)

EV_DIR = Path(EV_FILE).parent


class CLITest(TestCase):
    data_dir = Path(__file__).parent / "data"

    def setUp(self) -> None:
        super().setUp()
        self.runner = CliRunner()
        self.config_dir = Path(__file__).parent / "data" / "relative" / "config"
        self.commands = [
            "new-project",
            "train",
            "synthesize",
            "preprocess",
            "inspect-checkpoint",
        ]

    def wip_test_synthesize(self):
        # TODO: Here's a stub for getting synthesis unit tests working
        #       I believe we'll need to also pass a stats object to the created spec_model
        # TODO: add a test for making sure that `preprocessing` and `logs_and_checkpoints` folders don't get created.
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
            "1"  # Fallback for running tests on Mac
        )
        import random

        import torch

        # Set a manual seed, because some seeds cause the model
        # to fail to generate a proper wav file. This seed was taken
        # from running torch.seed() on a working run. Note: this is a bit
        # brittle, but this test is just to test that the synthesize command
        # works given two functional checkpoints. Further tests into the effects
        # of seeds should be looked into.
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(10719787423044995460)
        random.seed(10719787423044995460)
        vocoder = HiFiGAN(
            HiFiGANConfig.load_config_from_path(
                self.config_dir / f"{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml"
            )
        )
        spec_model = FastSpeech2(
            FastSpeech2Config.load_config_from_path(
                self.config_dir / f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml"
            ),
            lang2id={"default": 0},
            speaker2id={"default": 0},
            stats=Stats(
                pitch=StatsInfo(
                    min=150, max=300, std=2.0, mean=0.5, norm_max=1.0, norm_min=0.1
                ),
                energy=StatsInfo(
                    min=0.1, max=10.0, std=2.0, mean=0.5, norm_max=1.0, norm_min=0.1
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            vocoder_trainer = Trainer(default_root_dir=tmpdir_str, barebones=True)
            fp_trainer = Trainer(default_root_dir=tmpdir_str, barebones=True)
            vocoder_trainer.strategy.connect(vocoder)
            fp_trainer.strategy.connect(spec_model)
            fp_path = tmpdir / "fp.ckpt"
            fp_trainer.save_checkpoint(fp_path)
            vocoder_path = tmpdir / "vocoder.ckpt"
            vocoder_trainer.save_checkpoint(vocoder_path)
            with open(tmpdir / "utt.list", "w") as f:
                f.write(
                    "\n".join(
                        ["this is a test", "here is another test", "and a foo bar test"]
                    )
                )
            single_text_result = self.runner.invoke(
                app,
                [
                    "synthesize",
                    "from-text",
                    str(fp_path),
                    "--vocoder-path",
                    str(vocoder_path),
                    "-o",
                    str(tmpdir / "single_text"),
                    "--text",
                    "hello world",
                    "-O",
                    "wav",
                ],
            )
            self.assertEqual(single_text_result.exit_code, 0)
            self.assertEqual(
                len(list((tmpdir / "single_text" / "wav").glob("*.wav"))), 1
            )  # assert synthesizes a single file
            filelist_result = self.runner.invoke(
                app,
                [
                    "synthesize",
                    "from-text",
                    str(fp_path),
                    "--vocoder-path",
                    str(vocoder_path),
                    "-o",
                    str(tmpdir / "filelist"),
                    "--filelist",
                    str(tmpdir / "utt.list"),
                    "-O",
                    "wav",
                ],
            )
            self.assertEqual(filelist_result.exit_code, 0)
            self.assertEqual(
                len(list((tmpdir / "filelist" / "wav").glob("*.wav"))), 3
            )  # assert synthesizes three files

    def test_commands_present(self):
        result = self.runner.invoke(app, ["--help"])
        # each command has some help
        for command in self.commands:
            self.assertIn(command, result.stdout)
        # link to docs is present
        self.assertIn("https://docs.everyvoice.ca", result.stdout)

    def test_command_help_messages(self):
        for command in self.commands:
            result = self.runner.invoke(app, [command, "--help"])
            self.assertEqual(result.exit_code, 0)
            result = self.runner.invoke(app, [command, "-h"])
            self.assertEqual(result.exit_code, 0)

    def test_update_schema(self):
        dummy_contact = ContactInformation(
            contact_name="Test Runner", contact_email="info@everyvoice.ca"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            # Validate that schema generation works correctly.
            _ = self.runner.invoke(app, ["update-schemas", "-o", tmpdir])
            for filename, obj in SCHEMAS_TO_OUTPUT.items():
                with open(Path(tmpdir) / filename, encoding="utf8") as f:
                    schema = json.load(f)
                # serialize the model to json and then validate against the schema
                # Some objects will require a contact key
                try:
                    obj_instance = obj()
                except ValidationError:
                    obj_instance = obj(contact=dummy_contact)
                self.assertIsNone(
                    jsonschema.validate(
                        json.loads(obj_instance.model_dump_json()),
                        schema=schema,
                    )
                )

            # Make sure the generated schemas are identical to those saved in the repo,
            # i.e., that we didn't change the models but forget to update the schemas.
            for filename in SCHEMAS_TO_OUTPUT:
                with open(Path(tmpdir) / filename, encoding="utf8") as f:
                    new_schema = f.read()
                try:
                    with open(EV_DIR / ".schema" / filename, encoding="utf8") as f:
                        saved_schema = f.read()
                except FileNotFoundError:
                    raise AssertionError(
                        f'Schema file {filename} is missing, please run "everyvoice update-schemas".'
                    )
                self.assertEqual(
                    saved_schema,
                    new_schema,
                    'Schemas are out of date, please run "everyvoice update-schemas".',
                )

        # Next, but only if everything above passed, we make sure we can't overwrite
        # existing schemas by accident.
        result = self.runner.invoke(app, ["update-schemas"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("FileExistsError", str(result))

    def test_inspect_checkpoint_help(self):
        result = self.runner.invoke(app, ["inspect-checkpoint", "--help"])
        self.assertIn("inspect-checkpoint [OPTIONS] MODEL_PATH", result.stdout)

    def test_inspect_checkpoint(self):
        result = self.runner.invoke(
            app, ["inspect-checkpoint", str(self.data_dir / "test.ckpt")]
        )
        self.assertIn('global_step": 52256', result.stdout)
        self.assertIn(
            "We couldn't read your file, possibly because the version of EveryVoice that created it is incompatible with your installed version.",
            result.stdout,
        )
        self.assertIn("It appears to have 0.0 M parameters.", result.stdout)
        self.assertIn("Number of Parameters", result.stdout)

    def test_preprocessing_with_wrong_config(self):
        """
        The user should have a friendly message that informs them that they used the wrong config file type.
        """
        with capture_logs() as output:
            result = self.runner.invoke(
                app,
                [
                    "preprocess",
                    str(self.config_dir / "everyvoice-aligner.yaml"),
                ],
            )
            self.assertEqual(result.exit_code, 1)
            self.assertIn(
                "We are expecting a FastSpeech2Config but it looks like you provided a DFAlignerConfig",
                output[0],
            )

    def test_expensive_imports_are_tucked_away(self):
        """Make sure expensive imports are tucked away form the CLI help"""
        result = subprocess.run(
            ["everyvoice", "-h"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=dict(os.environ, PYTHONPROFILEIMPORTTIME="1"),
        )

        msg = '\n\nPlease avoid causing {} being imported from "everyvoice -h".\nIt is a relatively expensive import and slows down shell completion.\nRun "PYTHONPROFILEIMPORTTIME=1 everyvoice -h" and inspect the logs to see why it\'s being imported.'
        self.assertNotIn(b"shared_types", result.stderr, msg.format("shared_types.py"))
        self.assertNotIn(b"pydantic", result.stderr, msg.format("pydantic"))


class TestBaseCLIHelper(TestCase):
    def test_save_configuration_to_log_dir(self):
        with TemporaryDirectory() as tempdir, mute_logger(
            "everyvoice.base_cli.helpers"
        ):
            tempdir = Path(tempdir)
            config = FastSpeech2Config(
                contact=ContactInformation(
                    contact_name="Test Runner", contact_email="info@everyvoice.ca"
                ),
                training={
                    "logger": {
                        "save_dir": tempdir / "log",
                        "name": "unittest",
                    },
                },
            )
            save_configuration_to_log_dir(config)

            log_dir = config.training.logger.save_dir / config.training.logger.name
            log = log_dir / "log"
            self.assertTrue(log.exists())

            hparams = log_dir / "hparams.yaml"
            self.assertTrue(hparams.exists())
            with hparams.open(mode="r", encoding="UTF8") as f:
                config_reloaded = yaml.load(f, Loader=Loader)
                self.assertEqual(
                    config.training.logger.save_dir,
                    Path(config_reloaded["training"]["logger"]["save_dir"]),
                )
                self.assertEqual(
                    config.training.logger.name,
                    config_reloaded["training"]["logger"]["name"],
                )

    def test_complete_path_hack(self):
        from everyvoice.base_cli.interfaces import complete_path

        self.assertEqual(complete_path(), [])
