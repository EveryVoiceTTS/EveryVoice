#!/usr/bin/env python

import enum
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
import typer
import yaml
from packaging.version import Version
from pydantic import ValidationError
from pytest import main
from pytorch_lightning import Trainer
from typer.testing import CliRunner
from yaml import CLoader as Loader

import everyvoice.tests.model_stubs

# required for `./run_tests.py cli` to work, otherwise test_inspect_checkpoint
# fails with an Intel MKL FATAL ERROR saying it cannot load libtorch_cpu.so
import everyvoice.tests.test_model  # noqa
from everyvoice import __file__ as EV_FILE
from everyvoice._version import VERSION
from everyvoice.base_cli.helpers import save_configuration_to_log_dir
from everyvoice.cli import SCHEMAS_TO_OUTPUT, app
from everyvoice.config.shared_types import ContactInformation
from everyvoice.demo.app import create_demo_app
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
    capture_logs,
    capture_stdout,
    flatten_log,
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


class CLITest(TestCase):
    data_dir = Path(__file__).parent / "data"
    config_dir = Path(__file__).parent / "data" / "relative" / "config"
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
                    cls.config_dir / f"{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml"
                )
            )
            spec_model = FastSpeech2(
                FastSpeech2Config.load_config_from_path(
                    cls.config_dir / f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml"
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
        self.assertEqual(result.exit_code, 0)
        self.assertIn(VERSION, result.stdout)

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
        self.assertEqual(result.exit_code, 0)
        self.assertIn("EveryVoice version", result.stdout)
        self.assertIn("Python version", result.stdout)
        # We can't really validate the whole dependency list, but we should at least find torch
        # [5:] ignores the header generated by everyvoice --diagnostic and only looks at deps
        self.assertIn("torch", "".join(result.stdout.lower().splitlines()[5:]))

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
                self.assertEqual(single_text_result.exit_code, 0)
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
                with self.subTest(filename=filename, type=obj):
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
                with self.subTest(filename=filename):
                    with open(Path(tmpdir) / filename, encoding="utf8") as f:
                        new_schema = f.read().replace(
                            "\\\\", "/"
                        )  # force paths to posix
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

        # Next, but only if everything above passed, we make sure we can't overwrite
        # existing schemas by accident.
        result = self.runner.invoke(app, ["update-schemas"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("FileExistsError", str(result))

    def test_evaluate(self):
        result = self.runner.invoke(
            app,
            [
                "evaluate",
                "-f",
                self.data_dir / "LJ010-0008.wav",
                "-r",
                self.data_dir / "lj" / "wavs" / "LJ050-0269.wav",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("LJ010-0008", result.stdout)
        self.assertIn("STOI", result.stdout)
        self.assertIn("MOS", result.stdout)
        self.assertIn("SI-SDR", result.stdout)
        self.assertIn("PESQ", result.stdout)
        dir_result = self.runner.invoke(
            app,
            [
                "evaluate",
                "-d",
                self.data_dir / "lj" / "wavs",
                "-r",
                self.data_dir / "LJ010-0008.wav",
            ],
        )
        self.assertEqual(dir_result.exit_code, 0)
        self.assertIn("LJ050-0269", dir_result.stdout, "should print out the basenames")
        self.assertIn(
            "Average STOI",
            dir_result.stdout,
            "should report metrics in terms of averages",
        )
        evaluation_output = self.data_dir / "lj" / "wavs" / "evaluation.json"
        assert evaluation_output.exists(), "should print results to a file"
        evaluation_output.unlink()

    def test_old_inspect_checkpoint(self):
        result = self.runner.invoke(
            app, ["inspect-checkpoint", str(self.data_dir / "test.ckpt")]
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "This command has been renamed to `everyvoice checkpoint inspect`",
            flatten_log(result.stdout),
        )

    def test_inspect_checkpoint_help(self):
        result = self.runner.invoke(app, ["checkpoint", "inspect", "--help"])
        self.assertIn("checkpoint inspect [OPTIONS] MODEL_PATH", result.stdout)

    def test_inspect_checkpoint(self):
        result = self.runner.invoke(
            app, ["checkpoint", "inspect", str(self.data_dir / "test.ckpt")]
        )
        self.assertIn('global_step": 52256', result.stdout)
        self.assertIn(
            "We couldn't read your file, possibly because the version of EveryVoice that created it is incompatible with your installed version.",
            result.stdout,
        )
        self.assertIn("It appears to have 0.0 M parameters.", result.stdout)
        self.assertIn("Number of Parameters", result.stdout)

    def test_inspect_not_a_checkpoint(self) -> None:
        result = self.runner.invoke(app, ["checkpoint", "inspect", os.devnull])
        assert result.exit_code != 0
        assert "Error loading checkpoint" in str(result.exception)

    def test_inspect_good_fp_checkpoint(self) -> None:
        fp_path, _ = self.get_dummy_models()
        result = self.runner.invoke(app, ["checkpoint", "inspect", str(fp_path)])
        assert result.exit_code == 0
        assert "according to its model info: {'name': 'FastSpeech2'" in result.output
        assert "Trainable params" in result.output

    def test_inspect_good_vocoder_checkpoint(self) -> None:
        _, vocoder_path = self.get_dummy_models()
        result = self.runner.invoke(app, ["checkpoint", "inspect", str(vocoder_path)])
        assert result.exit_code == 0
        assert "according to its model info: {'name': 'HiFiGAN'" in result.output
        assert "Trainable params: 83,986,835" in result.output

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
            assert "HiFiGANGenerator" in result.output
            assert "Trainable params: 13,254,034" in result.output

    def test_preprocessing_with_wrong_config(self):
        """
        The user should have a friendly message that informs them that they used the wrong config file type.
        """
        with capture_logs() as output:
            result = self.runner.invoke(
                app,
                [
                    "preprocess",
                    str(self.config_dir / "everyvoice-spec-to-wav.yaml"),
                ],
            )
            self.assertEqual(result.exit_code, 1)
            self.assertIn(
                "We are expecting a FastSpeech2Config but it looks like you provided a HiFiGANConfig",
                "\n".join(output),
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

    def test_demo_with_bad_args(self):
        result = self.runner.invoke(app, ["demo"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Missing argument", result.output)

        result = self.runner.invoke(
            app, ["demo", os.devnull, os.devnull, "--output-format", "not-a-format"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid value", result.output)

    EMPTY_DEMO_ARGS: dict[str, Any] = {
        "languages": [],
        "speakers": [],
        "output_dir": None,
        "accelerator": None,
    }

    def test_create_demo_app_with_errors(self):
        # outputs is the first thing to get checked, because it can be done as
        # a quick check before loading any models.
        with self.assertRaises(ValueError) as cm:
            create_demo_app(
                text_to_spec_model_path=None,
                spec_to_wav_model_path=None,
                **self.EMPTY_DEMO_ARGS,  # type: ignore[arg-type]
                outputs=[],
            )
        self.assertIn("Empty outputs list", str(cm.exception))

        class WrongEnum(str, enum.Enum):
            foo = "foo"

        for outputs in (["wav", WrongEnum.foo], ["textgrid", "foo"]):
            with self.assertRaises(ValueError) as cm:
                create_demo_app(
                    text_to_spec_model_path=None,
                    spec_to_wav_model_path=None,
                    **self.EMPTY_DEMO_ARGS,  # type: ignore[arg-type]
                    outputs=outputs,
                )
            self.assertIn("Unknown output format 'foo'", str(cm.exception))

    def test_demo_with_bad_models(self) -> None:
        devnull = Path(os.devnull)
        with self.assertRaises(ValueError) as cm:
            create_demo_app(devnull, devnull, **self.EMPTY_DEMO_ARGS, outputs=["wav"])  # type: ignore[arg-type]
        self.assertIn("It does not appear to be a valid checkpoint", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            create_demo_app(
                devnull,
                self.data_dir / "test.ckpt",
                **self.EMPTY_DEMO_ARGS,  # type: ignore[arg-type]
                outputs=["wav"],
            )
        self.assertIn("maybe it's not actually a HiFiGAN model", str(cm.exception))

    def test_demo_with_wrong_models(self) -> None:
        fp_path, vocoder_path = self.get_dummy_models()
        with self.assertRaises(ValueError) as cm:
            create_demo_app(fp_path, fp_path, **self.EMPTY_DEMO_ARGS, outputs=["wav"])  # type: ignore[arg-type]
        self.assertIn("maybe it's not actually a HiFiGAN model", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            create_demo_app(
                vocoder_path,
                vocoder_path,
                **self.EMPTY_DEMO_ARGS,  # type: ignore[arg-type]
                outputs=["wav"],
            )
        self.assertIn("maybe it's not actually an fs2 model", str(cm.exception))

    def test_g2p(self):
        result = self.runner.invoke(
            app,
            [
                "g2p",
                "abc",
                str(self.data_dir / Path("text.txt")),
                "--config",
                str(self.config_dir / Path("everyvoice-shared-text.yaml")),
            ],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("['hello', 'world']", result.stdout)
        self.assertNotIn("['HELLO', 'WORLD']", result.stdout)

    def mock_create_demo_app(self, *_args, **_kwargs):
        class MockCreateDemoApp:
            def launch(self, *_args, **_kwargs):
                print(f"  - Launch Port: {_kwargs['server_port']}")
                print(f"  - Launch Share: {_kwargs['share']}")
                print(f"  - Launch Server Name: {_kwargs['server_name']}")
                if "config_file" in _kwargs:
                    print(f"  - Config File: {_kwargs['config_file']}")
                else:
                    print("  - Config File: None")

        return MockCreateDemoApp()

    def test_create_demo_app(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            # This test is just to make sure that the demo app can be created with parameters for gradio
            # and that it doesn't crash.
            _, vocoder_path = everyvoice.tests.model_stubs.get_stubbed_vocoder(
                tmpdir / "vocoder"
            )
            _, spec_model_path = everyvoice.tests.model_stubs.get_stubbed_model(
                tmpdir / "spec_model"
            )
            # This test is just to make sure that the demo app params are passed correctly
            port = 7000
            ip = "123.456.78.90"
            # with mock.patch(
            #    "everyvoice.demo.app.create_demo_app",
            #    side_effect=self.mock_create_demo_app,
            # ):
            with (
                mock.patch(
                    "everyvoice.demo.app.load_model_from_checkpoint",
                    side_effect=self.mock_demo_load_model_from_checkpoint,
                ),
                mock.patch(
                    "everyvoice.base_cli.helpers.inference_base_command",
                    side_effect=self.mock_fuction_placeholder,
                ),
                mock.patch(
                    "everyvoice.demo.app.synthesize_audio",
                    side_effect=self.mock_fuction_placeholder2,
                ),
                mock.patch(
                    "gradio.Blocks.launch",
                    return_value="Launching gradio app blocks",
                    side_effect=self.mock_fuction_placeholder2,
                ),
            ):
                result = self.runner.invoke(
                    app,
                    [
                        "demo",
                        str(spec_model_path),
                        str(vocoder_path),
                        "--port",
                        port,
                        "--share",
                        "--server-name",
                        ip,  # Mock IP address
                    ],
                )
            self.assertEqual(result.exit_code, 0)
            self.assertIn(f"  - Port: {port}", result.output)
            self.assertIn("  - Share: True", result.output)
            self.assertIn(f"  - Server Name: {ip}", result.output)

    def mock_demo_load_model_from_checkpoint(
        *_arg, **kwargs
    ) -> tuple:  # [FastSpeech2, torch.nn.Module, dict, torch.device]:
        print(
            "mock_demo_load_model_from_checkpoint called with args:",
            _arg,
            "and kwargs:",
            kwargs,
        )

        class model:
            from types import SimpleNamespace

            lang2id = {"default": 0}
            speaker2id = {"default": 0}
            model_data = {"use_global_style_token_module": False}
            config_data = {"model": SimpleNamespace(**model_data)}
            config = SimpleNamespace(
                **config_data,
            )

        return model, {}, {}, "cpu"  # Mock return values for the test

    def mock_fuction_placeholder(self, *args, **kwargs):
        """
        Mock function to replace any function that we are not testing.
        """
        print("mock_fuction_placeholder called with args:", args, "and kwargs:", kwargs)
        pass

    def mock_fuction_placeholder2(self, **kwargs):
        """
        Mock function to replace any function that we are not testing.
        """
        print("mock_fuction_placeholder2 called with kwargs:", kwargs)
        pass

    def test_create_demo_app_with_ui_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            # This test is just to make sure that the demo app can be created with parameters for gradio
            # and that it doesn't crash.
            _, vocoder_path = everyvoice.tests.model_stubs.get_stubbed_vocoder(
                tmpdir / "vocoder"
            )
            _, spec_model_path = everyvoice.tests.model_stubs.get_stubbed_model(
                tmpdir / "spec_model"
            )

            # Create a dummy app config file
            config: dict = {
                "app_title": "Test App",
                "app_description": "This is a test app description.",
                "app_instructions": "These are test app instructions.",
                "speakers": {
                    "default": "Person A",
                },
                "languages": {
                    "default": "English",
                },
                "input_text_label": "Input Text",
                "duration_multiplier_label": "Duration Multiplier",
                "language_label": "Language",
                "speaker_label": "Speaker",
                "output_format_label": "Output Format",
                "synthesize_label": "Synthesize",
                "file_output_label": "File Output",
            }
            config_file = tmpdir / "demo_config.json"
            with config_file.open("w", encoding="utf8") as f:
                json.dump(config, f)
            allowlist_file = tmpdir / "allowlist.txt"
            with allowlist_file.open("w", encoding="utf8") as f:
                f.write("hey\nyes\nword")
            # This test is just to make sure that the demo app params are passed correctly
            port = "7000"
            ip = "123.456.78.90"

            with (
                mock.patch(
                    "everyvoice.demo.app.load_model_from_checkpoint",
                    side_effect=self.mock_demo_load_model_from_checkpoint,
                ),
                mock.patch(
                    "everyvoice.base_cli.helpers.inference_base_command",
                    side_effect=self.mock_fuction_placeholder,
                ),
                mock.patch(
                    "everyvoice.demo.app.synthesize_audio",
                    side_effect=self.mock_fuction_placeholder2,
                ),
                mock.patch(
                    "gradio.Blocks.launch",
                    return_value="Launching gradio app blocks",
                    side_effect=self.mock_fuction_placeholder2,
                ),
            ):
                result = self.runner.invoke(
                    app,
                    [
                        "demo",
                        str(spec_model_path),
                        str(vocoder_path),
                        "--port",
                        port,
                        "--server-name",
                        ip,  # Mock IP address
                        "--ui-config-file",
                        str(config_file),
                        "--speaker",
                        "default",
                        "--language",
                        "default",
                        "--allowlist",
                        allowlist_file,
                    ],
                )
                # print(result.output, result.exit_code)  # Debug output

            self.assertEqual(result.exit_code, 0)
            self.assertIn(
                f"Using speakers from app config JSON: [('{config['speakers']['default']}', 'default')]",
                result.output,
            )
            self.assertIn(
                f"Using languages from app config JSON: [('{config['languages']['default']}', 'default')]",
                result.output,
            )

            self.assertIn(
                f"Using app title from app config JSON: {config['app_title']}",
                result.output,
            )

    def test_create_demo_app_with_malformed_ui_config_file(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            # This test is just to make sure that the demo app can be created with parameters for gradio
            # and that it doesn't crash.
            _, vocoder_path = everyvoice.tests.model_stubs.get_stubbed_vocoder(
                tmpdir / "vocoder"
            )
            _, spec_model_path = everyvoice.tests.model_stubs.get_stubbed_model(
                tmpdir / "spec_model"
            )
            # Create a malformed app config file (missing closing brace)
            config = """{
                "app_title": "Test App",
                "app_description": "This is a test app description.",
                "app_instructions": "These are test app instructions.",
                "input_text_label": "Input Text",
                "duration_multiplier_label": "Duration Multiplier",
                "language_label": "Language",
                "speaker_label": "Speaker",
                "output_format_label": "Output Format",
                "synthesize_label": "Synthesize",
                "file_output_label": "File Output",
            """
            config_file = tmpdir / "malformed_demo_config.json"
            with config_file.open("w", encoding="utf8") as f:
                f.write(config)
            # This test is just to make sure that the demo app params are passed correctly
            port = "7000"
            ip = "123.456.78.90"

            with (
                mock.patch(
                    "everyvoice.demo.app.load_model_from_checkpoint",
                    side_effect=self.mock_demo_load_model_from_checkpoint,
                ),
                mock.patch(
                    "everyvoice.base_cli.helpers.inference_base_command",
                    side_effect=self.mock_fuction_placeholder,
                ),
                mock.patch(
                    "everyvoice.demo.app.synthesize_audio",
                    side_effect=self.mock_fuction_placeholder2,
                ),
                mock.patch(
                    "gradio.Blocks.launch",
                    return_value="Launching gradio app blocks",
                    side_effect=self.mock_fuction_placeholder2,
                ),
            ):
                result = self.runner.invoke(
                    app,
                    [
                        "demo",
                        str(spec_model_path),
                        str(vocoder_path),
                        "--port",
                        port,
                        "--server-name",
                        ip,  # Mock IP address
                        "--ui-config-file",
                        str(config_file),
                        "--speaker",
                        "default",
                        "--language",
                        "default",
                    ],
                )
                # print(result.output, result.exit_code)  # Debug output
            self.assertNotEqual(result.exit_code, 0)
            self.assertRegex(
                result.output, r"(?s)Your config file.*malformed.*has.*errors"
            )

    # unit test for error handling in load_app_ui_labels
    def test_create_demo_load_app_ui_labels_errors(self):
        from everyvoice.demo.app import load_app_ui_labels

        # Create a dummy app config file
        config_bad_speaker = {
            "app_title": "Test App",
            "speakers": {
                "unknown": "Person A",
            },
            "languages": {
                "default": "English",
            },
        }
        config_bad_language = {
            "app_title": "Test App",
            "speakers": {
                "default": "Person A",
            },
            "languages": {
                "unknown": "English",
            },
        }

        with self.assertRaises(typer.BadParameter) as cm:
            load_app_ui_labels(
                config_bad_language,
                ["all"],
                ["all"],
                ["default"],
                ["default"],
            )
        self.assertIn(
            "The 'languages' key in the app config JSON does not match the languages provided.",
            str(cm.exception),
        )
        with self.assertRaises(typer.BadParameter) as cm:
            load_app_ui_labels(
                config_bad_speaker,
                ["all"],
                ["all"],
                ["default"],
                ["default"],
            )
        self.assertIn(
            "The 'speakers' key in the app config JSON does not match the speakers provided.",
            str(cm.exception),
        )
        with self.assertRaises(typer.BadParameter) as cm:
            load_app_ui_labels(
                config_bad_speaker,
                ["default"],
                ["unknown"],
                ["default"],
                ["default"],
            )

        self.assertIn(
            "Language option has been activated, but valid languages have not been provided. The model has been trained in ['default'] languages. Please select either 'all' or at least some of them.",
            str(cm.exception),
        )
        with self.assertRaises(typer.BadParameter) as cm:
            load_app_ui_labels(
                config_bad_speaker,
                ["unknown"],
                ["default"],
                ["default"],
                ["default"],
            )
        self.assertIn(
            "Speaker option has been activated, but valid speakers have not been provided. The model has been trained with ['default'] speakers. Please select either 'all' or at least some of them.",
            str(cm.exception),
        )

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

            with mock.patch("torch.save", side_effect=self.mock_fuction_placeholder):
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

                self.assertEqual(result.exit_code, 0)
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

            with mock.patch("torch.save", side_effect=self.mock_fuction_placeholder):
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
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("Speaker 'non_existing_speaker' not found", result.output)

    def test_rename_speaker_with_no_speakers(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            import torch

            tmpdir = Path(tmpdir_str)

            # Create an empty checkpoint file
            empty_ckpt: dict = {"hyper_parameters": {"speaker2id": {}}}
            torch.save(empty_ckpt, tmpdir / "empty.ckpt")

            with mock.patch("torch.save", side_effect=self.mock_fuction_placeholder):
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
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("No speakers found", result.output)


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
