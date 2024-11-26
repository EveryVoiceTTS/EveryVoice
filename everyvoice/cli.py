import json
import platform
import subprocess
import sys
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import Any, List, Optional

import typer
from rich import print as rich_print
from rich.panel import Panel

from everyvoice._version import VERSION
from everyvoice.base_cli.checkpoint import inspect as inspect_checkpoint
from everyvoice.base_cli.interfaces import complete_path
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
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.preprocess import (
    preprocess as preprocess_fs2,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.synthesize import (
    synthesize as synthesize_fs2,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.train import (
    train as train_fs2,
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
from everyvoice.utils import generic_psv_filelist_reader, spinner
from everyvoice.wizard import (
    ALIGNER_CONFIG_FILENAME_PREFIX,
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

        result = subprocess.run(["pip", "freeze"], capture_output=True, check=False)
        if result.returncode != 0 or result.stderr:
            print('Error running "pip freeze":')
            print(result.stderr.decode(), end="", file=sys.stderr)
        else:
            pip_freeze = result.stdout.decode().splitlines()
            print("\n*torch* modules installed using pip:")
            print("\n".join(module for module in pip_freeze if "torch" in module))
            print("\nOther modules installed using pip:")
            print("\n".join(module for module in pip_freeze if "torch" not in module))

        result = subprocess.run(["conda", "list"], capture_output=True, check=False)
        if result.returncode != 0 or result.stderr:
            # the installation probably didn't use conda, so just ignore this error
            pass
        else:
            print("\nModules installed using conda:")
            conda_list = result.stdout.decode().splitlines()[2:]
            print(
                "\n".join(
                    module
                    for module in conda_list
                    if "pypi" not in module and "<develop>" not in module
                )
            )

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
        shell_complete=complete_path,
    ),
    audio_directory: Optional[Path] = typer.Option(
        None,
        "--audio-directory",
        "-d",
        file_okay=False,
        dir_okay=True,
        help="The directory where multiple audio files are located for evaluation",
        shell_complete=complete_path,
    ),
    non_matching_reference: Optional[Path] = typer.Option(
        None,
        "--non-matching-reference",
        "-r",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to a Non Mathing Reference audio file, required for MOS prediction.",
        shell_complete=complete_path,
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
        shell_complete=complete_path,
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
@app.command(hidden=True)
def check_data(
    config_file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to your model configuration file.",
        shell_complete=complete_path,
    ),
    heavy_clip_detection: bool = typer.Option(False),
    heavy_objective_evaluation: bool = typer.Option(False),
):
    with spinner():
        from everyvoice.base_cli.helpers import MODEL_CONFIGS, load_unknown_config
        from everyvoice.config.preprocessing_config import PreprocessingConfig
        from everyvoice.preprocessor import Preprocessor

    config = load_unknown_config(config_file)
    if not any((isinstance(config, x) for x in MODEL_CONFIGS)):
        print(
            "Sorry, your file does not appear to be a valid model configuration. Please choose another model config file."
        )
        sys.exit(1)
    assert not isinstance(config, PreprocessingConfig)
    training_filelist = generic_psv_filelist_reader(config.training.training_filelist)
    val_filelist = generic_psv_filelist_reader(config.training.validation_filelist)
    combined_filelist_data = training_filelist + val_filelist
    preprocessor = Preprocessor(config)
    checked_data = preprocessor.check_data(
        filelist=combined_filelist_data,
        heavy_clip_detection=heavy_clip_detection,
        heavy_objective_evaluation=heavy_objective_evaluation,
    )
    if not combined_filelist_data:
        print(
            f"Sorry, the data at {config.training.training_filelist} and {config.training.validation_filelist} is empty so there is nothing to check."
        )
        sys.exit(1)
    else:
        with open("checked-data.json", "w", encoding="utf8") as f:
            json.dump(checked_data, f)


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


class TestSuites(str, Enum):
    all = "all"
    cli = "cli"
    config = "config"
    dev = "dev"
    model = "model"
    preprocessing = "preprocessing"
    text = "text"
    fs2 = "fs2"


@app.command(hidden=True)
def test(suite: TestSuites = typer.Argument(TestSuites.dev)):
    """Run a test suite"""
    from everyvoice.run_tests import run_tests

    run_tests(suite)


# Deferred full initialization to optimize the CLI, but still exposed for unit testing.
SCHEMAS_TO_OUTPUT: dict[str, Any] = {}  # dict[str, type[BaseModel]]


@app.command()
def demo(
    text_to_spec_model: Path = typer.Argument(
        ...,
        file_okay=True,
        exists=True,
        dir_okay=False,
        help="The path to a trained text-to-spec (i.e., feature prediction) EveryVoice model.",
        shell_complete=complete_path,
    ),
    spec_to_wav_model: Path = typer.Argument(
        ...,
        help="The path to a trained vocoder.",
        dir_okay=False,
        file_okay=True,
        shell_complete=complete_path,
    ),
    allowlist: Path = typer.Option(
        None,
        "--allowlist",
        file_okay=True,
        dir_okay=False,
        help="A plain text file containing a list of words or utterances to allow synthesizing. Words/utterances should be separated by a new line in a plain text file. All other words are disallowed.",
        shell_complete=complete_path,
    ),
    denylist: Path = typer.Option(
        None,
        "--denylist",
        file_okay=True,
        dir_okay=False,
        help="A plain text file containing a list of words or utterances to disallow synthesizing. Words/utterances should be separated by a new line in a plain text file. All other words are allowed. IMPORTANT: there are many ways to 'hack' the denylist that we do not protect against. We suggest using the 'allowlist' instead for maximum security if you know the full list of utterances you want to allow synthesis for.",
        shell_complete=complete_path,
    ),
    languages: List[str] = typer.Option(
        ["all"],
        "--language",
        "-l",
        help="Specify languages to be included in the demo. Example: everyvoice demo <path_to_text_to_spec_model> <path_to_spec_to_wav_model> --language eng --language fin",
    ),
    speakers: List[str] = typer.Option(
        ["all"],
        "--speaker",
        "-s",
        help="Specify speakers to be included in the demo. Example: everyvoice demo <path_to_text_to_spec_model> <path_to_spec_to_wav_model> --speaker speaker_1 --speaker Sue",
    ),
    output_dir: Path = typer.Option(
        "synthesis_output",
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="The directory where your synthesized audio should be written",
        shell_complete=complete_path,
    ),
    accelerator: str = typer.Option("auto", "--accelerator", "-a"),
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

    with spinner():
        from everyvoice.demo.app import create_demo_app

        demo = create_demo_app(
            text_to_spec_model_path=text_to_spec_model,
            spec_to_wav_model_path=spec_to_wav_model,
            languages=languages,
            speakers=speakers,
            output_dir=output_dir,
            accelerator=accelerator,
            allowlist=allowlist_data,
            denylist=denylist_data,
        )

    demo.launch()


@app.command(hidden=True)
def update_schemas(
    out_dir: Path = typer.Option(
        None,
        "-o",
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        exists=True,
        shell_complete=complete_path,
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
    from everyvoice.model.aligner.config import AlignerConfig
    from everyvoice.model.e2e.config import EveryVoiceConfig
    from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
    from everyvoice.model.vocoder.config import VocoderConfig

    # We should not be changing the schema for patches, so only include major/minor version
    MAJOR_MINOR_VERSION = ".".join(VERSION.split(".")[:2])

    SCHEMAS_TO_OUTPUT.update(
        {
            f"{ALIGNER_CONFIG_FILENAME_PREFIX}-{MAJOR_MINOR_VERSION}.json": AlignerConfig,
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


CLICK_APP = typer.main.get_group(app)

if __name__ == "__main__":
    app()
