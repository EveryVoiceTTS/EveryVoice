import enum
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import typer
from pytest import raises
from typer.testing import CliRunner

from everyvoice.cli import app
from everyvoice.demo.app import create_demo_app
from everyvoice.tests.stubs import (
    TEST_DATA_DIR,
    flatten_log,
    mock_function_placeholder,
    mock_function_placeholder2,
)


def test_demo_with_bad_args():
    # No checkpoint → help message
    result = CliRunner().invoke(app, ["demo"])
    # Exit code for no-arg-is-help is 0 with click<=8.1.8 and typer<=0.23.2,
    # 2 if either is more recent
    assert result.exit_code in (0, 2)
    # the runner calls "root" instead of "everyvoice"
    assert "Usage: root demo [OPTIONS] CHECKPOINT" in flatten_log(result.output)

    # Invalid --output-format value
    result = CliRunner().invoke(
        app,
        [
            "demo",
            os.devnull,
            "--vocoder",
            os.devnull,
            "--output-format",
            "not-a-format",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid value" in flatten_log(result.output)


EMPTY_DEMO_ARGS: dict[str, Any] = {
    "languages": [],
    "speakers": [],
    "output_dir": None,
    "accelerator": None,
}


def test_create_demo_app_with_errors():
    # outputs is the first thing to get checked, because it can be done as
    # a quick check before loading any models.
    with raises(ValueError, match="Empty outputs list"):
        create_demo_app(
            text_to_spec_model_path=None,
            spec_to_wav_model_path=None,
            **EMPTY_DEMO_ARGS,  # type: ignore[arg-type]
            outputs=[],
        )

    class WrongEnum(str, enum.Enum):
        foo = "foo"

    for outputs in (["wav", WrongEnum.foo], ["textgrid", "foo"]):
        with raises(ValueError, match="Unknown output format 'foo'"):
            create_demo_app(
                text_to_spec_model_path=None,
                spec_to_wav_model_path=None,
                **EMPTY_DEMO_ARGS,  # type: ignore[arg-type]
                outputs=outputs,
            )


def test_demo_with_bad_models() -> None:
    devnull = Path(os.devnull)
    with raises(ValueError, match="It does not appear to be a valid checkpoint"):
        create_demo_app(devnull, devnull, **EMPTY_DEMO_ARGS, outputs=["wav"])  # type: ignore[arg-type]

    with raises(ValueError, match="maybe it's not actually a HiFiGAN model"):
        create_demo_app(
            devnull,
            TEST_DATA_DIR / "test.ckpt",
            **EMPTY_DEMO_ARGS,  # type: ignore[arg-type]
            outputs=["wav"],
        )


def test_demo_with_wrong_models(stubbed_model, stubbed_vocoder) -> None:
    _, fp_path = stubbed_model
    _, vocoder_path = stubbed_vocoder
    with raises(ValueError, match="maybe it's not actually a HiFiGAN model"):
        create_demo_app(fp_path, fp_path, **EMPTY_DEMO_ARGS, outputs=["wav"])  # type: ignore[arg-type]

    with raises(ValueError, match="maybe it's not actually an fs2 model"):
        create_demo_app(
            vocoder_path,
            vocoder_path,
            **EMPTY_DEMO_ARGS,  # type: ignore[arg-type]
            outputs=["wav"],
        )


def mock_create_demo_app(*_args, **_kwargs):
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


def test_create_demo_app(stubbed_model, stubbed_vocoder):
    # This test is just to make sure that the demo app can be created with parameters for gradio
    # and that it doesn't crash.
    _, vocoder_path = stubbed_vocoder
    _, spec_model_path = stubbed_model
    # This test is just to make sure that the demo app params are passed correctly
    port = 7000
    ip = "123.456.78.90"
    # with mock.patch(
    #    "everyvoice.demo.app.create_demo_app",
    #    side_effect=mock_create_demo_app,
    # ):
    with (
        mock.patch(
            "everyvoice.cli._peek_model_class",
            return_value="FastSpeech2",
        ),
        mock.patch(
            "everyvoice.demo.app.load_model_from_checkpoint",
            side_effect=mock_demo_load_model_from_checkpoint,
        ),
        mock.patch(
            "everyvoice.base_cli.helpers.inference_base_command",
            side_effect=mock_function_placeholder,
        ),
        mock.patch(
            "everyvoice.demo.app.synthesize_audio",
            side_effect=mock_function_placeholder2,
        ),
        mock.patch(
            "gradio.Blocks.launch",
            return_value="Launching gradio app blocks",
            side_effect=mock_function_placeholder2,
        ),
    ):
        result = CliRunner().invoke(
            app,
            [
                "demo",
                str(spec_model_path),
                "--vocoder",
                str(vocoder_path),
                "--port",
                port,
                "--share",
                "--server-name",
                ip,  # Mock IP address
            ],
        )
    assert result.exit_code == 0
    assert f"Port: {port}" in flatten_log(result.output)
    assert "Share: True" in flatten_log(result.output)
    assert f"Server Name: {ip}" in flatten_log(result.output)


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


def test_create_demo_app_with_ui_config_file(stubbed_model, stubbed_vocoder) -> None:
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        # This test is just to make sure that the demo app can be created with parameters for gradio
        # and that it doesn't crash.
        _, vocoder_path = stubbed_vocoder
        _, spec_model_path = stubbed_model

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
                "everyvoice.cli._peek_model_class",
                return_value="FastSpeech2",
            ),
            mock.patch(
                "everyvoice.demo.app.load_model_from_checkpoint",
                side_effect=mock_demo_load_model_from_checkpoint,
            ),
            mock.patch(
                "everyvoice.base_cli.helpers.inference_base_command",
                side_effect=mock_function_placeholder,
            ),
            mock.patch(
                "everyvoice.demo.app.synthesize_audio",
                side_effect=mock_function_placeholder2,
            ),
            mock.patch(
                "gradio.Blocks.launch",
                return_value="Launching gradio app blocks",
                side_effect=mock_function_placeholder2,
            ),
        ):
            result = CliRunner().invoke(
                app,
                [
                    "demo",
                    str(spec_model_path),
                    "--vocoder",
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

        assert result.exit_code == 0
        assert (
            f"Using speakers from app config JSON: [('{config['speakers']['default']}', 'default')]"
            in result.output
        )
        assert (
            f"Using languages from app config JSON: [('{config['languages']['default']}', 'default')]"
            in result.output
        )

        assert (
            f"Using app title from app config JSON: {config['app_title']}"
            in result.output
        )


def test_create_demo_app_with_malformed_ui_config_file(stubbed_model, stubbed_vocoder):
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        # This test is just to make sure that the demo app can be created with parameters for gradio
        # and that it doesn't crash.
        _, vocoder_path = stubbed_vocoder
        _, spec_model_path = stubbed_model
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
                "everyvoice.cli._peek_model_class",
                return_value="FastSpeech2",
            ),
            mock.patch(
                "everyvoice.demo.app.load_model_from_checkpoint",
                side_effect=mock_demo_load_model_from_checkpoint,
            ),
            mock.patch(
                "everyvoice.base_cli.helpers.inference_base_command",
                side_effect=mock_function_placeholder,
            ),
            mock.patch(
                "everyvoice.demo.app.synthesize_audio",
                side_effect=mock_function_placeholder2,
            ),
            mock.patch(
                "gradio.Blocks.launch",
                return_value="Launching gradio app blocks",
                side_effect=mock_function_placeholder2,
            ),
        ):
            result = CliRunner().invoke(
                app,
                [
                    "demo",
                    str(spec_model_path),
                    "--vocoder",
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
        assert result.exit_code != 0
        assert re.search(r"(?s)Your config file.*malformed.*has.*errors", result.output)


# unit test for error handling in load_app_ui_labels
def test_create_demo_load_app_ui_labels_errors():
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

    with raises(
        typer.BadParameter,
        match="The 'languages' key in the app config JSON does not match the languages provided.",
    ):
        load_app_ui_labels(
            config_bad_language,
            ["all"],
            ["all"],
            ["default"],
            ["default"],
        )
    with raises(
        typer.BadParameter,
        match="The 'speakers' key in the app config JSON does not match the speakers provided.",
    ):
        load_app_ui_labels(
            config_bad_speaker,
            ["all"],
            ["all"],
            ["default"],
            ["default"],
        )
    with raises(
        typer.BadParameter,
        match=r"Language option has been activated, but valid languages have not been provided. The model has been trained in \['default'\] languages. Please select either 'all' or at least some of them.",
    ):
        load_app_ui_labels(
            config_bad_speaker,
            ["default"],
            ["unknown"],
            ["default"],
            ["default"],
        )

    with raises(
        typer.BadParameter,
        match=r"Speaker option has been activated, but valid speakers have not been provided. The model has been trained with \['default'\] speakers. Please select either 'all' or at least some of them.",
    ):
        load_app_ui_labels(
            config_bad_speaker,
            ["unknown"],
            ["default"],
            ["default"],
            ["default"],
        )


def test_demo_dispatch_styletts2_rejects_vocoder_flag():
    """Passing --vocoder with a StyleTTS2 checkpoint should produce a clear error."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        import torch

        tmpdir = Path(tmpdir_str)
        fake_ckpt = tmpdir / "styletts2.ckpt"
        torch.save(
            {
                "model_info": {"name": "StyleTTS2Module"},
                "hyper_parameters": {"mode": "second", "config": {}},
                "state_dict": {},
            },
            fake_ckpt,
        )
        fake_vocoder = tmpdir / "hifigan.ckpt"
        fake_vocoder.touch()

        result = CliRunner().invoke(
            app,
            [
                "demo",
                str(fake_ckpt),
                "--vocoder",
                str(fake_vocoder),
                "--ref-speaker",
                f"Eric={fake_ckpt}",  # reuse fake_ckpt as a dummy audio file
            ],
        )
        assert result.exit_code != 0
        assert "StyleTTS2 does not use a separate vocoder" in flatten_log(result.output)


def test_demo_dispatch_fs2_requires_vocoder(stubbed_model):
    """Invoking demo with a FastSpeech2 checkpoint but no --vocoder should error."""
    _, spec_model_path = stubbed_model

    with mock.patch(
        "everyvoice.cli._peek_model_class",
        return_value="FastSpeech2",
    ):
        result = CliRunner().invoke(
            app,
            ["demo", str(spec_model_path)],
        )
    assert result.exit_code != 0
    assert "FastSpeech2 requires a vocoder checkpoint" in flatten_log(result.output)


def test_demo_dispatch_fs2_rejects_ref_speaker(stubbed_model, stubbed_vocoder):
    """Passing --ref-speaker with a FastSpeech2 checkpoint should produce a clear error."""
    _, vocoder_path = stubbed_vocoder
    _, spec_model_path = stubbed_model

    with mock.patch(
        "everyvoice.cli._peek_model_class",
        return_value="FastSpeech2",
    ):
        result = CliRunner().invoke(
            app,
            [
                "demo",
                str(spec_model_path),
                "--vocoder",
                str(vocoder_path),
                "--ref-speaker",
                f"Eric={spec_model_path}",
            ],
        )
    assert result.exit_code != 0
    assert "--ref-speaker is only used with StyleTTS2" in flatten_log(result.output)


def test_demo_dispatch_vocoder_checkpoint_as_primary():
    """Passing a HiFiGAN checkpoint as the primary CHECKPOINT should give a helpful error."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        import torch

        tmpdir = Path(tmpdir_str)
        fake_vocoder_ckpt = tmpdir / "hifigan.ckpt"
        torch.save(
            {"model_info": {"name": "HiFiGAN"}, "state_dict": {}},
            fake_vocoder_ckpt,
        )

        result = CliRunner().invoke(
            app,
            ["demo", str(fake_vocoder_ckpt)],
        )
        assert result.exit_code != 0
        assert "appears to be a standalone vocoder checkpoint" in flatten_log(
            result.output
        )
