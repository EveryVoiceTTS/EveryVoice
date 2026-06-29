#!/usr/bin/env python

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase, mock

import jsonschema
import yaml
from packaging.version import Version
from pydantic import ValidationError
from pytest import main
from pytorch_lightning import Trainer
from typer.testing import CliRunner
from yaml import CLoader as Loader

# required for `./run_tests.py cli` to work, otherwise test_inspect_checkpoint
# fails with an Intel MKL FATAL ERROR saying it cannot load libtorch_cpu.so
import everyvoice.tests.test_model  # noqa
from everyvoice import __file__ as EV_FILE
from everyvoice._version import VERSION
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
from everyvoice.tests.stubs import (
    TEST_DATA_DIR,
    capture_logs,
    capture_stdout,
    flatten_log,
    mock_function_placeholder,
    temp_chdir,
)
from everyvoice.wizard import (
    SPEC_TO_WAV_CONFIG_FILENAME_PREFIX,
    TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX,
)

EV_DIR = Path(EV_FILE).parent


def major_minor(version):
    v = Version(version)
    return f"{v.major}.{v.minor}"


CONFIG_DIR = TEST_DATA_DIR / "relative" / "config"


class CLITest(TestCase):
    class_tmp_dir_obj: Any
    class_tmp_dir: str
    dummy_fp_path: Path
    dummy_vocoder_path: Path

    def setUp(self) -> None:
        super().setUp()
        self.runner = CliRunner()
        self.commands = [
            "new-project",
            "train",
            "synthesize",
            "preprocess",
            "checkpoint",
            "evaluate",
            "demo",
            "g2p",
        ]

    @classmethod
    def setUpClass(cls):
        cls.class_tmp_dir_obj = tempfile.TemporaryDirectory(
            prefix="ev_test_cli", dir="."
        )
        cls.class_tmp_dir = cls.class_tmp_dir_obj.name

    @classmethod
    def tearDownClass(cls):
        cls.class_tmp_dir_obj.cleanup()
        if hasattr(cls, "dummy_fp_path"):
            del cls.dummy_fp_path
        if hasattr(cls, "dummy_vocoder_path"):
            del cls.dummy_vocoder_path

    @classmethod
    def get_dummy_models(cls) -> tuple[Path, Path]:
        """Usage: dummy_fp_path, dummy_vocoder_path = self.get_dummy_models()"""
        if not hasattr(cls, "dummy_fp_path"):
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
                    CONFIG_DIR / f"{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml"
                )
            )
            spec_model = FastSpeech2(
                FastSpeech2Config.load_config_from_path(
                    CONFIG_DIR / f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml"
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
            tmpdir_str = cls.class_tmp_dir
            tmpdir = Path(tmpdir_str)
            vocoder_trainer = Trainer(default_root_dir=tmpdir_str, barebones=True)
            fp_trainer = Trainer(default_root_dir=tmpdir_str, barebones=True)
            vocoder_trainer.strategy.connect(vocoder)
            fp_trainer.strategy.connect(spec_model)
            cls.dummy_fp_path = tmpdir / "fp.ckpt"
            fp_trainer.save_checkpoint(cls.dummy_fp_path)
            cls.dummy_vocoder_path = tmpdir / "vocoder.ckpt"
            vocoder_trainer.save_checkpoint(cls.dummy_vocoder_path)
            os.system(f"ls -la {tmpdir_str}")

        return (cls.dummy_fp_path, cls.dummy_vocoder_path)

    def test_version(self):
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert VERSION in result.stdout

    def test_submodule_versions(self):
        # Team decision 2025-02-10: we won't keep submodule versions in complete lockstep,
        # but we'll match major.minor.
        # But don't check wav2vec2aligner at all, it's much more independent.
        from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2._version import (
            VERSION as FS2_VERSION,
        )
        from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl._version import (
            VERSION as HFGL_VERSION,
        )

        self.assertEqual(
            major_minor(VERSION),
            major_minor(FS2_VERSION),
            "please keep FastSpeech2_lightning and EveryVoice major.minor verion in sync",
        )
        self.assertEqual(
            major_minor(VERSION),
            major_minor(HFGL_VERSION),
            "please keep HiFiGAN_iSTFT_lightning and EveryVoice major.minor version in sync",
        )

    def test_diagnostic(self):
        with capture_stdout():
            result = self.runner.invoke(app, ["--diagnostic"])
        assert result.exit_code == 0
        assert "EveryVoice version" in result.stdout
        assert "Python version" in result.stdout
        # We can't really validate the whole dependency list, but we should at least find torch
        # [5:] ignores the header generated by everyvoice --diagnostic and only looks at deps
        assert "torch" in "".join(result.stdout.lower().splitlines()[5:])

    def wip_test_synthesize(self):
        # TODO: Here's a stub for getting synthesis unit tests working
        #       I believe we'll need to also pass a stats object to the created spec_model
        # TODO: add a test for making sure that `preprocessing` and `logs_and_checkpoints` folders don't get created.
        # 20260428 update: this test works on Linux and Windows, but not on MacOS. TODO: fix it on MacOS
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
            "1"  # Fallback for running tests on Mac
        )

        fp_path, vocoder_path = self.get_dummy_models()
        fp_path = fp_path.resolve()
        vocoder_path = vocoder_path.resolve()
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            with temp_chdir(tmpdir):
                with open("utt.list", "w", encoding="utf8") as f:
                    f.write(
                        "\n".join(
                            [
                                "this is a test",
                                "here is another test",
                                "and a foo bar test",
                            ]
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
                        str("single_text"),
                        "--text",
                        "hello world",
                        "-O",
                        "wav",
                    ],
                )
                assert single_text_result.exit_code == 0
                self.assertEqual(
                    len(list(Path("single_text/wav").glob("*.wav"))), 1
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
                        "filelist",
                        "--filelist",
                        "utt.list",
                        "-O",
                        "wav",
                    ],
                )
                assert filelist_result.exit_code == 0
                self.assertEqual(
                    len(list((tmpdir / "filelist" / "wav").glob("*.wav"))), 3
                )  # assert synthesizes three files

    def test_commands_present(self):
        result = self.runner.invoke(app, ["--help"])
        # each command has some help
        for command in self.commands:
            assert command in result.stdout
        # link to docs is present
        assert "https://docs.everyvoice.ca" in result.stdout

    def test_command_help_messages(self):
        for command in self.commands:
            result = self.runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0
            result = self.runner.invoke(app, [command, "-h"])
            assert result.exit_code == 0

    def test_update_schemas(self):
        dummy_contact = ContactInformation(
            contact_name="Test Runner", contact_email="info@everyvoice.ca"
        )
        with tempfile.TemporaryDirectory() as tmpdir_s:
            tmpdir = Path(tmpdir_s)
            # Validate that schema generation works correctly.
            _ = self.runner.invoke(app, ["update-schemas", "-o", tmpdir_s])
            for filename, obj in SCHEMAS_TO_OUTPUT.items():
                with self.subTest(filename=filename, type=obj):
                    with open(tmpdir / filename, encoding="utf8") as f:
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
                with self.subTest(filename=filename):
                    with open(tmpdir / filename, encoding="utf8") as f:
                        new_schema = f.read()
                    try:
                        with open(EV_DIR / ".schema" / filename, encoding="utf8") as f:
                            saved_schema = f.read()
                    except FileNotFoundError as e:
                        raise AssertionError(
                            f'Schema file {filename} is missing, please run "everyvoice update-schemas".'
                        ) from e
                    self.assertEqual(
                        saved_schema,
                        new_schema,
                        'Schemas are out of date, please run "everyvoice update-schemas".',
                    )

            # Make sure we can't overwrite existing but out-of-date schemas by accident.
            with open(tmpdir / next(iter(SCHEMAS_TO_OUTPUT)), "w") as f:
                # Make one of the schemas forcefully out of date
                f.write("asdf")
            result = self.runner.invoke(app, ["update-schemas", "-o", tmpdir_s])
            assert result.exit_code != 0
            assert "ERROR" in flatten_log(result.output)
            assert "Out of date" in flatten_log(result.output)

        # If everything above passed, running update-schemas should say schemas are up to date
        result = self.runner.invoke(app, ["update-schemas"])
        assert result.exit_code == 0
        assert "already up to date" in flatten_log(result.output)
        assert "Out of date" not in flatten_log(result.output)
        assert "ERROR" not in flatten_log(result.output)

    def test_evaluate(self):
        result = self.runner.invoke(
            app,
            [
                "evaluate",
                "-f",
                TEST_DATA_DIR / "LJ010-0008.wav",
                "-r",
                TEST_DATA_DIR / "lj" / "wavs" / "LJ050-0269.wav",
            ],
        )
        assert result.exit_code == 0
        assert "LJ010-0008" in result.stdout
        assert "STOI" in result.stdout
        assert "MOS" in result.stdout
        assert "SI-SDR" in result.stdout
        assert "PESQ" in result.stdout
        dir_result = self.runner.invoke(
            app,
            [
                "evaluate",
                "-d",
                TEST_DATA_DIR / "lj" / "wavs",
                "-r",
                TEST_DATA_DIR / "LJ010-0008.wav",
            ],
        )
        assert dir_result.exit_code == 0
        assert "LJ050-0269", dir_result.stdout in "should print out the basenames"
        self.assertIn(
            "Average STOI",
            dir_result.stdout,
            "should report metrics in terms of averages",
        )
        evaluation_output = TEST_DATA_DIR / "lj" / "wavs" / "evaluation.json"
        assert evaluation_output.exists(), "should print results to a file"
        evaluation_output.unlink()

    def test_old_inspect_checkpoint(self):
        result = self.runner.invoke(
            app, ["inspect-checkpoint", str(TEST_DATA_DIR / "test.ckpt")]
        )
        assert result.exit_code == 0
        self.assertIn(
            "This command has been renamed to `everyvoice checkpoint inspect`",
            flatten_log(result.stdout),
        )

    def test_inspect_checkpoint_help(self):
        result = self.runner.invoke(app, ["checkpoint", "inspect", "--help"])
        assert "checkpoint inspect [OPTIONS] MODEL_PATH" in result.stdout

    def test_inspect_checkpoint(self):
        result = self.runner.invoke(
            app, ["checkpoint", "inspect", str(TEST_DATA_DIR / "test.ckpt")]
        )
        assert 'global_step": 52256' in result.stdout
        self.assertIn(
            "We couldn't read your file, possibly because the version of EveryVoice that created it is incompatible with your installed version.",
            result.stdout,
        )
        assert "It appears to have 0.0 M parameters." in result.stdout
        assert "Number of Parameters" in result.stdout

    def test_inspect_not_a_checkpoint(self) -> None:
        result = self.runner.invoke(app, ["checkpoint", "inspect", os.devnull])
        assert result.exit_code != 0
        assert "Error loading checkpoint" in str(result.exception)

    def test_inspect_good_fp_checkpoint(self) -> None:
        fp_path, _ = self.get_dummy_models()
        result = self.runner.invoke(app, ["checkpoint", "inspect", str(fp_path)])
        assert result.exit_code == 0
        assert "according to its model info: {'name': 'FastSpeech2'" in flatten_log(
            result.output
        )
        assert "Trainable params" in flatten_log(result.output)

    def test_inspect_good_vocoder_checkpoint(self) -> None:
        _, vocoder_path = self.get_dummy_models()
        result = self.runner.invoke(app, ["checkpoint", "inspect", str(vocoder_path)])
        assert result.exit_code == 0
        assert "according to its model info: {'name': 'HiFiGAN'" in flatten_log(
            result.output
        )
        assert "Trainable params: 83,986,835" in flatten_log(result.output)

    def test_export_and_inspect_generator(self) -> None:
        _, vocoder_path = self.get_dummy_models()
        with tempfile.TemporaryDirectory(prefix="generator_", dir=".") as tmpdir_str:
            exported_path = Path(tmpdir_str) / "exported.ckpt"
            result = self.runner.invoke(
                app,
                ["export", "spec-to-wav", "-o", str(exported_path), str(vocoder_path)],
            )
            assert result.exit_code == 0
            assert exported_path.exists()

            result = self.runner.invoke(
                app, ["checkpoint", "inspect", str(exported_path)]
            )
            assert result.exit_code == 0
            assert "HiFiGANGenerator" in flatten_log(result.output)
            assert "Trainable params: 13,254,034" in flatten_log(result.output)

    def test_preprocessing_with_wrong_config(self):
        """
        The user should have a friendly message that informs them that they used the wrong config file type.
        """
        with capture_logs() as output:
            result = self.runner.invoke(
                app,
                [
                    "preprocess",
                    "text-to-spec",
                    str(CONFIG_DIR / "everyvoice-spec-to-wav.yaml"),
                ],
            )
            assert result.exit_code == 1
            self.assertIn(
                "We are expecting a FastSpeech2Config but it looks like you provided a HiFiGANConfig",
                "\n".join(output),
            )

    def test_preprocess_without_subcommand_shows_subcommands(self):
        """'everyvoice preprocess' without a subcommand shows available subcommands."""
        result = self.runner.invoke(app, ["preprocess"])
        # no_args_is_help=True causes the help text to list both subcommands
        assert "text-to-spec" in flatten_log(result.output)
        assert "text-to-wav" in flatten_log(result.output)

    def test_preprocess_text_to_wav_help(self):
        """'everyvoice preprocess text-to-wav --help' should exit cleanly."""
        result = self.runner.invoke(app, ["preprocess", "text-to-wav", "--help"])
        # Exit code for no-arg-is-help is 0 with click<=8.1.8 and typer<=0.23.2,
        # 2 if either is more recent
        assert result.exit_code in (0, 2)
        assert (
            "preprocess text-to-wav [OPTIONS] CONFIG_FILE"
            in flatten_log(result.output).strip()
        )

    def test_expensive_imports_are_tucked_away(self):
        """Make sure expensive imports are tucked away form the CLI help"""
        result = subprocess.run(
            ["everyvoice", "-h"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=dict(os.environ, PYTHONPROFILEIMPORTTIME="1"),
            check=True,
        )

        msg = '\n\nPlease avoid causing {} being imported from "everyvoice -h".\nIt is a relatively expensive import and slows down shell completion.\nRun "PYTHONPROFILEIMPORTTIME=1 everyvoice -h" and inspect the logs to see why it\'s being imported.'
        self.assertNotIn(b"shared_types", result.stderr, msg.format("shared_types.py"))
        self.assertNotIn(b"pydantic", result.stderr, msg.format("pydantic"))

    def test_g2p(self):
        result = self.runner.invoke(
            app,
            [
                "g2p",
                "abc",
                str(TEST_DATA_DIR / "text.txt"),
                "--config",
                str(CONFIG_DIR / "everyvoice-shared-text.yaml"),
            ],
        )

        assert result.exit_code == 0
        assert "['hello', 'world']" in result.stdout
        self.assertNotIn("['HELLO', 'WORLD']", result.stdout)

    def test_rename_speaker(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            import torch

            tmpdir = Path(tmpdir_str)
            # Create a test checkpoint file with speakers
            ckpt = {
                "hyper_parameters": {
                    "speaker2id": {"old_speaker": 0, "another_speaker": 1}
                }
            }
            torch.save(ckpt, tmpdir / "test.ckpt")

            with mock.patch("torch.save", side_effect=mock_function_placeholder):
                # Mock the torch.save function to avoid writing files
                result = self.runner.invoke(
                    app,
                    [
                        "checkpoint",
                        "rename-speaker",
                        str(tmpdir / "test.ckpt"),
                        "old_speaker",
                        "new_speaker",
                    ],
                )

                assert result.exit_code == 0
                self.assertIn(
                    "Renamed speaker 'old_speaker' to 'new_speaker'.", result.output
                )
                self.assertIn(
                    "Updated speakers: {'another_speaker': 1, 'new_speaker': 0}",
                    result.output,
                )

    def test_rename_speaker_with_non_existing_speaker(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            import torch

            tmpdir = Path(tmpdir_str)
            # Create a test checkpoint file with speakers
            ckpt = {
                "hyper_parameters": {
                    "speaker2id": {"old_speaker": 0, "another_speaker": 1}
                }
            }
            torch.save(ckpt, tmpdir / "test.ckpt")

            with mock.patch("torch.save", side_effect=mock_function_placeholder):
                # Test renaming a non-existing speaker
                result = self.runner.invoke(
                    app,
                    [
                        "checkpoint",
                        "rename-speaker",
                        str(tmpdir / "test.ckpt"),
                        "non_existing_speaker",
                        "new_speaker",
                    ],
                )
                # print(result.output)
                assert result.exit_code != 0
                assert "Speaker 'non_existing_speaker' not found" in flatten_log(
                    result.output
                )

    def test_rename_speaker_with_no_speakers(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            import torch

            tmpdir = Path(tmpdir_str)

            # Create an empty checkpoint file
            empty_ckpt: dict = {"hyper_parameters": {"speaker2id": {}}}
            torch.save(empty_ckpt, tmpdir / "empty.ckpt")

            with mock.patch("torch.save", side_effect=mock_function_placeholder):
                # Test renaming with no speakers in the checkpoint
                result = self.runner.invoke(
                    app,
                    [
                        "checkpoint",
                        "rename-speaker",
                        str(tmpdir / "empty.ckpt"),
                        "old_speaker",
                        "new_speaker",
                    ],
                )
                # print(result.output)
                assert result.exit_code != 0
                assert "No speakers found" in flatten_log(result.output)


class TestBaseCLIHelper(TestCase):
    def test_save_configuration_to_log_dir(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tempdir_s:
            tempdir = Path(tempdir_s)
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
            assert log.exists()

            hparams = log_dir / "hparams.yaml"
            assert hparams.exists()
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


if __name__ == "__main__":
    main(sys.argv)
