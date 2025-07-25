import json
import platform
import subprocess
import sys
from enum import Enum
from io import TextIOWrapper
from pathlib import Path
from textwrap import dedent
from typing import Annotated, Any, List, Optional

import typer
from merge_args import merge_args
from rich import print as rich_print
from rich.panel import Panel

from everyvoice._version import VERSION
from everyvoice.base_cli.checkpoint import inspect as inspect_checkpoint
from everyvoice.base_cli.checkpoint import rename_speaker
from everyvoice.base_cli.interfaces import inference_base_command_interface
from everyvoice.model.aligner.wav2vec2aligner.aligner.cli import (
    ALIGN_SINGLE_LONG_HELP,
    ALIGN_SINGLE_SHORT_HELP,
    CLI_LONG_HELP,
    EXTRACT_SEGMENTS_LONG_HELP,
    EXTRACT_SEGMENTS_SHORT_HELP,
)
from everyvoice.model.aligner.wav2vec2aligner.aligner.cli import (
    align_single as ctc_segment_align_single,
)
from everyvoice.model.aligner.wav2vec2aligner.aligner.cli import (
    extract_segments_from_textgrid,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.check_data import (
    check_data_command,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.preprocess import (
    preprocess as preprocess_fs2,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.synthesize import (
    synthesize as synthesize_fs2,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.train import (
    train as train_fs2,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions import (
    SynthesizeOutputFormats,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import (
    HFG_EXPORT_LONG_HELP,
    HFG_EXPORT_SHORT_HELP,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import (
    export as export_hfg,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import (
    synthesize as synthesize_hfg,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import train as train_hfg
from everyvoice.run_tests import SUITE_NAMES, run_tests
from everyvoice.utils import spinner
from everyvoice.wizard import (
    PREPROCESSING_CONFIG_FILENAME_PREFIX,
    SPEC_TO_WAV_CONFIG_FILENAME_PREFIX,
    TEXT_CONFIG_FILENAME_PREFIX,
    TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX,
    TEXT_TO_WAV_CONFIG_FILENAME_PREFIX,
)


# See https://github.com/tiangolo/typer/issues/428#issuecomment-1238866548
class TyperGroupOrderAsDeclared(typer.core.TyperGroup):
    def list_commands(self, ctx):
        return self.commands.keys()


app = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
    cls=TyperGroupOrderAsDeclared,
    help="""
    # Welcome to the EveryVoice Command Line Interface

    Please visit the documentation to get started: [https://docs.everyvoice.ca](https://docs.everyvoice.ca)

    ## Use your own language/dataset

    To run the wizard for a new project please use the following command: everyvoice new-project

    ## Segment long files in your dataset

    If you have long audio files that contain more than one utterance,
    you can use the segmentation tool by running everyvoice segment [OPTIONS]

    ## Preprocess

    Once you have a configuration, preprocess your data by running everyvoice preprocess [OPTIONS]

    ## Check Data

    You can optionally check your data by running everyvoice check-data [OPTIONS]

    ## Train

    Once you have a configuration and have preprocessed your data, train a model by running everyvoice train [text-to-spec|spec-to-wav] [OPTIONS].
    EveryVoice has different types of models you can train:

    1. **text-to-spec** --- this is the most common model you will need to train. It is a model from text inputs to spectral feature (aka spectrogram) outputs.

    2. **spec-to-wav** --- this is the model that turns your spectral features into audio. It is also known as a 'vocoder'. You will typically not need to train your own version. Please refer to [https://pathtocheckpoints](https://pathtocheckpoints) for more information.

    ## Synthesize

    Once you have a trained model, generate some audio by running: everyvoice synthesize [text-to-spec|spec-to-wav] [OPTIONS]

    ## Evaluate

    You can also try to evaluate your model by running: everyvoice evaluate [synthesized_audio.wav|folder_containing_wavs] [OPTIONS]

""",
)


def _diagnostic_pip(use_uv: bool) -> bool:
    """
    Try to list installed packages using `uv` or `pip`.
    """
    try:
        result = subprocess.run(
            (["uv"] if use_uv else ["python", "-m"]) + ["pip", "freeze"],
            capture_output=True,
            check=False,
        )
        # if "EveryVoice" not in result.stdout.decode():
        #     return False
        if result.returncode == 0 and (use_uv or not result.stderr):
            # result.stderr contains: "Using Python 3.11.10 environment at venv"
            pip_freeze = result.stdout.decode().splitlines()
            print("\n*torch* modules installed using pip:")
            print("\n".join(module for module in pip_freeze if "torch" in module))
            print("\nOther modules installed using pip:")
            print("\n".join(module for module in pip_freeze if "torch" not in module))

        return True
    except FileNotFoundError:
        return False


def _diagnostic_conda() -> None:
    """
    Try to list installed packages using `conda`.
    """
    try:
        result = subprocess.run(["conda", "list"], capture_output=True, check=False)
        print("Environment type: conda")
        if result.returncode == 0 and not result.stderr:
            print("\nModules installed using conda:")
            conda_list = result.stdout.decode().splitlines()
            print(
                "\n".join(
                    module
                    for module in conda_list
                    if "pypi" not in module and "<develop>" not in module
                )
            )
    except FileNotFoundError:
        # the installation probably didn't use conda, so just ignore this error
        pass


@app.callback(invoke_without_command=True)
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", help="Print the version of EveryVoice and exit."
    ),
    diagnostic: Optional[bool] = typer.Option(
        None, "--diagnostic", "-d", help="Print diagnostic information and exit."
    ),
):
    """The top-level function that gets called first"""
    if version:
        print(VERSION)
        sys.exit(0)
    if diagnostic:
        print("EveryVoice Diagnostic information")
        print(f"EveryVoice version: {VERSION}")
        print(f"Python version: {sys.version}")
        uname = platform.uname()
        print(f"System: {uname.system} {uname.release} {uname.version} {uname.machine}")

        # Do `uv` first then use `pip` which covers both `pip` and `conda` envs.
        _diagnostic_conda()
        if not (_diagnostic_pip(use_uv=True) or _diagnostic_pip(use_uv=False)):
            print("Unable to get installed package versions")

        sys.exit(0)


@app.command(
    short_help="Evaluate your synthesized audio",
    name="evaluate",
    help="""
    # Evalution help

    This command will evaluate an audio file, or a folder containing multiple audio files. Currently this is done by calculating the metrics from Kumar et. al. 2023.
    We will report the predicted Wideband Perceptual Estimation of Speech Quality (PESQ), Short-Time Objective Intelligibility (STOI), and Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) by default.
    We will also report the estimation of subjective Mean Opinion Score (MOS) if a Non-Matching Reference is provided. Please refer to Kumar et. al. for more information.



    Kumar, Anurag, et al. “TorchAudio-Squim: Reference-less Speech Quality and Intelligibility measures in TorchAudio.” ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023.
    """,
)
def evaluate(
    audio_file: Optional[Path] = typer.Option(
        None,
        "--audio-file",
        "-f",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to a single audio file for evaluation.",
    ),
    audio_directory: Optional[Path] = typer.Option(
        None,
        "--audio-directory",
        "-d",
        file_okay=False,
        dir_okay=True,
        help="The directory where multiple audio files are located for evaluation",
    ),
    non_matching_reference: Optional[Path] = typer.Option(
        None,
        "--non-matching-reference",
        "-r",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to a Non Mathing Reference audio file, required for MOS prediction.",
    ),
):
    from tabulate import tabulate
    from tqdm import tqdm

    from everyvoice.evaluation import (
        calculate_objective_metrics_from_single_path,
        calculate_subjective_metrics_from_single_path,
        load_squim_objective_model,
        load_squim_subjective_model,
    )

    HEADERS = ["BASENAME", "STOI", "PESQ", "SI-SDR"]

    objective_model, o_sr = load_squim_objective_model()
    if non_matching_reference:
        subjective_model, s_sr = load_squim_subjective_model()
        HEADERS.append("MOS")

    if audio_file and audio_directory:
        print(
            "Sorry, please choose to evaluate either a single file or an entire directory. Got values for both."
        )
        sys.exit(1)

    def calculate_row(single_audio):
        stoi, pesq, si_sdr = calculate_objective_metrics_from_single_path(
            single_audio, objective_model, o_sr
        )
        row = [single_audio.stem, stoi, pesq, si_sdr]
        if non_matching_reference:
            mos = calculate_subjective_metrics_from_single_path(
                single_audio, non_matching_reference, subjective_model, s_sr
            )
            row.append(mos)
        return row

    if audio_file:
        row = calculate_row(audio_file)
        rich_print(
            Panel(
                tabulate([row], HEADERS, tablefmt="simple"),
                title=f"Objective Metrics for {audio_file}:",
            )
        )
        sys.exit(0)

    if audio_directory:
        import numpy as np

        results = []
        for wav_file in tqdm(
            audio_directory.glob("*.wav"),
            desc=f"Evaluating filies in {audio_directory}",
        ):
            row = calculate_row(wav_file)
            results.append(row)
        rich_print(
            Panel(
                tabulate(results, HEADERS, tablefmt="simple"),
                title=f"Objective Metrics for files in {audio_directory}:",
            )
        )
        arr = np.asarray(results)
        arr_float = arr[:, 1:].astype(np.float16)
        # remove nans
        arr_float = arr_float[
            ~np.isnan(arr_float).any(axis=1)
        ]  # ignore basename and check if any of the other values are nans
        n_metrics = arr_float.shape[1]
        mean_results = [arr_float[:, x].mean() for x in range(n_metrics)]
        std_results = [arr_float[:, x].std() for x in range(n_metrics)]
        avg_results = [
            f"{format(mean_results[x], '.4f')} ± {format(std_results[x], '.4f')}"
            for x in range(n_metrics)
        ]
        rich_print(
            Panel(
                tabulate(
                    [avg_results],
                    [f"Average {x}" for x in HEADERS[1:]],
                    tablefmt="simple",
                ),
                title=f"Average Objective Metrics for files in {audio_directory}:",
            )
        )
        print(f"Printing results to {audio_directory / 'evaluation.json'}")
        with open(audio_directory / "evaluation.json", "w") as f:
            json.dump(results, f)


class ModelTypes(str, Enum):
    text_to_spec = "text-to-spec"
    spec_to_wav = "spec-to-wav"


# Add the export commands
export_group = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
    cls=TyperGroupOrderAsDeclared,
    help="""
    # Export Help

        - **spec-to-wav** --- You can export your spec-to-wav model to a much smaller format for synthesis. Advanced: this will export only the generator, leaving the weights of the discriminators behind.
    """,
)

export_group.command(
    short_help=HFG_EXPORT_SHORT_HELP,
    name="spec-to-wav",
    help=HFG_EXPORT_LONG_HELP,
)(export_hfg)

app.add_typer(
    export_group,
    name="export",
    short_help="Export your EveryVoice models",
)

# Add the segment commands
segment_group = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
    cls=TyperGroupOrderAsDeclared,
    help=CLI_LONG_HELP,
)


segment_group.command(
    short_help=ALIGN_SINGLE_SHORT_HELP,
    name="align",
    help=ALIGN_SINGLE_LONG_HELP,
)(ctc_segment_align_single)

segment_group.command(
    short_help=EXTRACT_SEGMENTS_SHORT_HELP,
    name="extract",
    help=EXTRACT_SEGMENTS_LONG_HELP,
)(extract_segments_from_textgrid)

app.add_typer(
    segment_group,
    name="segment",
    short_help="Align and segment audio",
)


@app.command(
    short_help="This command will help you create all the configuration necessary for using a new project.",
    help="""
    # This command will help you create all the configuration necessary for using a new project.

    ## Getting Started

    In order to get started, please:

        1. make sure your audio data is available on your computer and in .wav format

        2. have a 'metadata' file that minimally has two columns, one for the text of the audio and one for the basename of the file


    ## Extra details and examples

    Inside /path/to/wavs/ you should have wav audio files like test0001.wav, test0002.wav etc - they can be called anything you want but the part of the file (minus the .wav portion) must be in your metadata file.

    Example wavs directory: "/path/to/wavs/"

    Example metadata: [https://github.com/EveryVoiceTTS/EveryVoice/blob/main/everyvoice/filelists/lj_full.psv](https://github.com/EveryVoiceTTS/EveryVoice/blob/main/everyvoice/filelists/lj_full.psv)

""",
)
def new_project(
    trace: bool = typer.Option(
        False, help="Enable question tree trace mode.", hidden=True
    ),
    debug_state: bool = typer.Option(
        False, help="Enable wizard state debug/trace mode.", hidden=True
    ),
    resume_from: Optional[Path] = typer.Option(
        None,
        "--resume-from",
        "-r",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="Resume from previously saved progress.",
    ),
):
    from everyvoice.wizard.main_tour import get_main_wizard_tour

    rich_print(
        Panel(
            dedent(
                """
                    Welcome to the EveryVoice Wizard. We will guide you through the process of setting up the configuration for a new EveryVoice project.

                    Navigation: as any point, you can hit Ctrl+C to: go back a step, view progress, save progress, or exit the wizard.

                    From saved progress, you can resume at any time by running the same command with the --resume-from option.
                """
            ).strip()
        )
    )

    get_main_wizard_tour(trace=trace, debug_state=debug_state).run(
        resume_from=resume_from
    )


# Add preprocess to root
app.command(
    short_help="Preprocess your data",
    help=f"""
    # Preprocess Help

    This command will preprocess all of the data you need for use with EveryVoice.

    By default every step of the preprocessor will be done by running:
    \n\n
    **everyvoice preprocess config/{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml**
    \n\n
    If you only want to process specific things, you can run specific commands by adding them as options for example:
    \n\n
    **everyvoice preprocess config/{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml -s energy -s pitch**
    """,
)(preprocess_fs2)


# Add check_data to root
app.command(
    "check-data",
    short_help="Check your data for outliers or any anomalies",
    help="""
    # Check Data Help

    This command will check all of your data to help you find anomalies and outliers.

    To check your data, make sure you've run preprocessing first (everyvoice preprocess --help).
    Then you need to briefly and partially train a text-to-spec model. We recommend 100-1000 steps to start.

    Then, with your partially trained model you can run the data checker:
    \n\n
    **everyvoice check-data config/everyvoice-text-to-spec.yaml logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt**
    \n\n

    This will output two files - one containing some basic statistics for your data and the other containing losses for each datapoint as calculated by your model.

    """,
)(check_data_command)


# Add the train commands
train_group = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
    cls=TyperGroupOrderAsDeclared,
    help="""
    # Train Help

        - **text-to-spec** --- this is the most common model you will need to train. It is a model from text inputs to spectral feature (aka spectrogram) outputs.

        - **spec-to-wav** --- this is the model that turns your spectral features into audio. It is also known as a 'vocoder'. You will typically not need to train your own version. Please refer to [https://pathtocheckpoints](https://pathtocheckpoints) for more information.
    """,
)

train_group.command(
    name="text-to-spec",
    short_help="Train your Text-to-Spec model",
    help=f"""Train your text-to-spec model.  For example:

    **everyvoice train text-to-spec config/{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml**
    """,
)(train_fs2)

train_group.command(
    name="spec-to-wav",
    short_help="Train your Spec-to-Wav model",
    help=f"""Train your spec-to-wav model.  For example:

    **everyvoice train spec-to-wav config/{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml**
    """,
)(train_hfg)

app.add_typer(
    train_group,
    name="train",
    short_help="Train your EveryVoice models",
)

# Add synthesize commands
synthesize_group = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
    cls=TyperGroupOrderAsDeclared,
    help="""
    # Synthesize Help

        - **from-text** --- This is the most common input for performing normal speech synthesis. It will take text or a filelist with text and produce either waveform audio or spectrogram.

        - **from-spec** --- This is the model that turns your spectral features into audio. This type of synthesis is also known as copy synthesis and unless you know what you are doing, you probably don't want to do this.
    """,
)

synthesize_group.command(
    name="from-text",
)(synthesize_fs2)

synthesize_group.command(
    name="from-spec",
)(synthesize_hfg)

app.add_typer(
    synthesize_group,
    name="synthesize",
    short_help="Synthesize using your pre-trained EveryVoice models",
)

app.command(
    name="inspect-checkpoint",
    short_help="Extract structural information from a checkpoint",
)(inspect_checkpoint)

app.command(
    name="rename-speaker",
    short_help="Rename a speaker in the checkpoint's parameters",
)(rename_speaker)


TestSuites = Enum("TestSuites", {name: name for name in SUITE_NAMES})  # type: ignore


@app.command(hidden=True)
def test(suite: TestSuites = typer.Argument("dev")):  # pragma: no cover
    """Run a test suite"""
    try:
        import everyvoice.tests  # noqa: F401

        run_tests(suite.value)
    except ModuleNotFoundError:
        print(
            "ERROR: hidden command 'everyvoice test' only works when you install EveryVoice from source.",
            file=sys.stderr,
        )


# Deferred full initialization to optimize the CLI, but still exposed for unit testing.
SCHEMAS_TO_OUTPUT: dict[str, Any] = {}  # dict[str, type[BaseModel]]


AllowedDemoOutputFormats = Enum(  # type: ignore
    "AllowedDemoOutputFormats",
    [("all", "all")] + [(i.name, i.value) for i in SynthesizeOutputFormats],
)


@app.command()
@merge_args(inference_base_command_interface)
def demo(
    text_to_spec_model: Path = typer.Argument(
        ...,
        file_okay=True,
        exists=True,
        dir_okay=False,
        help="The path to a trained text-to-spec (i.e., feature prediction) EveryVoice model.",
    ),
    spec_to_wav_model: Path = typer.Argument(
        ...,
        help="The path to a trained vocoder.",
        dir_okay=False,
        file_okay=True,
    ),
    allowlist: Path = typer.Option(
        None,
        "--allowlist",
        file_okay=True,
        dir_okay=False,
        help="A plain text file containing a list of words or utterances to allow synthesizing. Words/utterances should be separated by a new line in a plain text file. All other words are disallowed.",
    ),
    denylist: Path = typer.Option(
        None,
        "--denylist",
        file_okay=True,
        dir_okay=False,
        help="A plain text file containing a list of words or utterances to disallow synthesizing. Words/utterances should be separated by a new line in a plain text file. All other words are allowed. IMPORTANT: there are many ways to 'hack' the denylist that we do not protect against. We suggest using the 'allowlist' instead for maximum security if you know the full list of utterances you want to allow synthesis for.",
    ),
    languages: List[str] = typer.Option(
        ["all"],
        "--language",
        "-l",
        help="Specify languages to be included in the demo. Must be supported by your model. Example: everyvoice demo TEXT_TO_SPEC_MODEL SPEC_TO_WAV_MODEL --language eng --language fin",
    ),
    speakers: List[str] = typer.Option(
        ["all"],
        "--speaker",
        "-s",
        help="Specify speakers to be included in the demo. Must be supported by your model. Example: everyvoice demo TEXT_TO_SPEC_MODEL SPEC_TO_WAV_MODEL --speaker speaker_1 --speaker Sue",
    ),
    outputs: list[AllowedDemoOutputFormats] = typer.Option(
        ["all"],
        "--output-format",
        "-O",
        help="Specify output formats to be included in the demo. Example: everyvoice demo TEXT_TO_SPEC_MODEL SPEC_TO_WAV_MODEL --output-format wav --output-format readalong-html",
    ),
    output_dir: Path = typer.Option(
        "synthesis_output",
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="The directory where your synthesized audio should be written",
    ),
    accelerator: str = typer.Option(
        "auto",
        "--accelerator",
        "-a",
        help="Specify the Pytorch Lightning accelerator to use",
    ),
    port: int = typer.Option(7860, "--port", "-p", help="The port to run the demo on."),
    share: bool = typer.Option(
        False,
        "--share",
        help="Share the demo using Gradio's share feature. This will make the demo accessible from the internet.",
    ),
    server_name: str = typer.Option(
        "0.0.0.0",
        "--server-name",
        "-n",
        help="The server name to run the demo on. This is useful if you want to run the demo on a specific IP address.",
    ),
    **kwargs,
):
    if allowlist and denylist:
        raise ValueError(
            "You provided a value for both the allowlist and the denylist but you can only provide one."
        )

    allowlist_data = []
    denylist_data = []

    if allowlist:
        with open(allowlist) as f:
            allowlist_data = [x.strip() for x in f]

    if denylist:
        with open(denylist) as f:
            denylist_data = [x.strip() for x in f]

    # print the parameters to the console
    print("INFO - Starting the demo with the following parameters:")
    print(f"  - Text-to-Spec Model: {text_to_spec_model}")
    print(f"  - Spec-to-Wav Model: {spec_to_wav_model}")
    print(f"  - Languages: {languages}")
    print(f"  - Speakers: {speakers}")
    print(f"  - Outputs: {outputs}")
    print(f"  - Output Directory: {output_dir}")
    print(f"  - Accelerator: {accelerator}")
    print(f"  - Allowlist: {allowlist_data if allowlist else 'None'}")
    print(f"  - Denylist: {denylist_data if denylist else 'None'}")
    print(f"  - Port: {port}")
    print(f"  - Share: {share}")
    print(f"  - Server Name: {server_name}")

    with spinner():
        from everyvoice.demo.app import create_demo_app

        demo = create_demo_app(
            text_to_spec_model_path=text_to_spec_model,
            spec_to_wav_model_path=spec_to_wav_model,
            languages=languages,
            speakers=speakers,
            outputs=outputs,
            output_dir=output_dir,
            accelerator=accelerator,
            allowlist=allowlist_data,
            denylist=denylist_data,
            **kwargs,
        )

    demo.launch(share=share, server_port=port, server_name=server_name)


@app.command(hidden=True)
def update_schemas(
    out_dir: Path = typer.Option(
        None,
        "-o",
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        exists=True,
    ),
):
    """Update the JSON Schemas. This is hidden because you shouldn't be calling this unless you are
    a developer for EveryVoice. Note: Pydantic will raise some Warnings related to the Callable fields
    having string Schemas. These can be ignored.
    """
    if out_dir is None:
        schema_dir_path = Path(__file__).parent / ".schema"
    else:
        schema_dir_path = out_dir

    # Defer somewhat slow imports to optimize CLI
    from everyvoice.config.preprocessing_config import PreprocessingConfig
    from everyvoice.config.text_config import TextConfig
    from everyvoice.model.e2e.config import EveryVoiceConfig
    from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
    from everyvoice.model.vocoder.config import VocoderConfig

    # We should not be changing the schema for patches, so only include major/minor version
    MAJOR_MINOR_VERSION = ".".join(VERSION.split(".")[:2])

    SCHEMAS_TO_OUTPUT.update(
        {
            f"{TEXT_TO_WAV_CONFIG_FILENAME_PREFIX}-{MAJOR_MINOR_VERSION}.json": EveryVoiceConfig,
            f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}-{MAJOR_MINOR_VERSION}.json": FeaturePredictionConfig,
            f"{PREPROCESSING_CONFIG_FILENAME_PREFIX}-{MAJOR_MINOR_VERSION}.json": PreprocessingConfig,
            f"{TEXT_CONFIG_FILENAME_PREFIX}-{MAJOR_MINOR_VERSION}.json": TextConfig,
            f"{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}-{MAJOR_MINOR_VERSION}.json": VocoderConfig,
        }
    )

    for filename, schema in SCHEMAS_TO_OUTPUT.items():
        if (schema_dir_path / filename).exists():
            raise FileExistsError(
                f"Sorry a schema already exists for version {filename}.\n"
                "If it's already been published to the schema store, please bump the EveryVoice minor version number and generate the schemas again.\n"
                "If the current minor version is still in development, just delete the schema files and try again."
            )
        with open(schema_dir_path / filename, "w", encoding="utf8") as f:
            json.dump(schema.model_json_schema(), f, indent=2)
            f.write("\n")


@app.command()
def g2p(
    lang_id: Annotated[str, typer.Argument(help="lang id")],
    # Ignoring mypy since class FileText(io.TextIOWrapper)
    input_file: Annotated[typer.FileText, typer.Argument()] = TextIOWrapper(
        sys.stdin.buffer,
        encoding="utf-8",
    ),  # type: ignore[assignment]
    config: Annotated[
        Optional[Path],
        typer.Option(help="full to path to everyvoice-shared-text.yaml"),
    ] = None,
):
    """
    Apply G2P to stdin.
    Great for testing your EveryVoice g2p plugin.
    """
    from everyvoice.config.text_config import TextConfig
    from everyvoice.text.phonemizer import AVAILABLE_G2P_ENGINES as G2Ps
    from everyvoice.text.phonemizer import get_g2p_engine

    if config:
        text_config: TextConfig = TextConfig.load_config_from_path(config)
        print(
            f"Config contains custon G2P Engines: {text_config.g2p_engines}",
            file=sys.stderr,
        )

    print("g2p available languages:", G2Ps.keys(), file=sys.stderr)
    g2p = get_g2p_engine(lang_id)
    for line in map(str.strip, input_file):
        print(g2p(line))


CLICK_APP = typer.main.get_group(app)

if __name__ == "__main__":
    app()
