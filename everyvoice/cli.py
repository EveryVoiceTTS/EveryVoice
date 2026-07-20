import platform
import subprocess
import sys
from enum import Enum
from io import TextIOWrapper
from pathlib import Path
from textwrap import dedent
from typing import Annotated, Any, Optional

import typer
from merge_args import merge_args
from rich import print as rich_print
from rich.panel import Panel

from everyvoice._version import VERSION
from everyvoice.base_cli import command, default_typer_args
from everyvoice.base_cli.checkpoint import inspect, load_checkpoint, rename_speaker
from everyvoice.base_cli.interfaces import (
    inference_base_command_interface,
    typer_directory_option,
    typer_file_argument,
    typer_file_option,
)
from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel
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
from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.cli.fetch_pretrained import (
    fetch_pretrained as fetch_pretrained_styletts2,
)
from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.cli.preprocess import (
    preprocess as preprocess_styletts2,
)
from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.cli.synthesize import (
    synthesize as synthesize_styletts2,
)
from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.cli.train import (
    train as train_styletts2,
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
    preprocess as preprocess_hfg,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import (
    synthesize as synthesize_hfg,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.cli import train as train_hfg
from everyvoice.wizard import (
    PREPROCESSING_CONFIG_FILENAME_PREFIX,
    SPEC_TO_WAV_CONFIG_FILENAME_PREFIX,
    TEXT_CONFIG_FILENAME_PREFIX,
    TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX,
    TEXT_TO_WAV_CONFIG_FILENAME_PREFIX,
)


# For the main `everyvoice` command, TyperGroupOrderAsDeclared doesn't work because
# single commands gets listed first, followed by sub-command groups. There are so many
# top-level commands and command groups now that it gets quite confusing, so let's order
# them logically for the main ones, and alphabetically for the rest.
class MainCommandOrder(typer.core.TyperGroup):
    def list_commands(self, ctx):
        # Order will be these first, then the rest alphabetically
        order = (
            "new-project",
            "preprocess",
            "train",
            "synthesize",
            "demo",
        )
        order_d = {x: i for i, x in enumerate(order)}
        return sorted(self.commands.keys(), key=lambda x: (order_d.get(x, 100), x))


app = typer.Typer(
    **{**default_typer_args, "cls": MainCommandOrder},
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

    Once you have a configuration and have preprocessed your data, train a model by running everyvoice train [text-to-spec|spec-to-wav|text-to-wav] [OPTIONS].
    EveryVoice has different types of models you can train:

    1. **text-to-spec** --- this is the most common model you will need to train. It is a model from text inputs to spectral feature (aka spectrogram) outputs.

    2. **spec-to-wav** --- this is the model that turns your spectral features into audio. It is also known as a 'vocoder'. You will typically not need to train your own version. Please refer to [https://pathtocheckpoints](https://pathtocheckpoints) for more information.

    3. **text-to-wav** --- this is a model that directly trains text input to waveform/audio output. It is much more heavy duty than training a text-to-spec model. A text-to-spec model will take less than 1 day to train for most datasets, while a text-to-wav model will take over a week for most datasets. For that reason, we strongly recommend training a text-to-wav model first. If you are satisfied with the results, and have extra GPU resources, you can try training a text-to-wav model for improved sentence-level audio quality.

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
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-v", help="Print the version of EveryVoice and exit."
        ),
    ] = False,
    diagnostic: Annotated[
        bool,
        typer.Option(
            "--diagnostic", "-d", help="Print diagnostic information and exit."
        ),
    ] = False,
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


@command(
    app,
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
    audio_file: Annotated[
        Optional[Path],
        typer_file_option(
            "--audio-file",
            "-f",
            help="The path to a single audio file for evaluation.",
        ),
    ] = None,
    audio_directory: Annotated[
        Optional[Path],
        typer_directory_option(
            "--audio-directory",
            "-d",
            help="The directory where multiple audio files are located for evaluation",
        ),
    ] = None,
    non_matching_reference: Annotated[
        Optional[Path],
        typer_file_option(
            "--non-matching-reference",
            "-r",
            help="The path to a Non Mathing Reference audio file, required for MOS prediction.",
        ),
    ] = None,
):
    import json

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

    if (audio_file and audio_directory) or (
        audio_file is None and audio_directory is None
    ):
        print(
            "Sorry, please choose to evaluate either a single file or an entire directory."
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
    **default_typer_args,
    help="""
    # Export Help

        - **spec-to-wav** --- You can export your spec-to-wav model to a much smaller format for synthesis. Advanced: this will export only the generator, leaving the weights of the discriminators behind.
    """,
)

command(
    export_group,
    short_help=HFG_EXPORT_SHORT_HELP,
    name="spec-to-wav",
    help=HFG_EXPORT_LONG_HELP,
)(export_hfg)

app.add_typer(
    export_group, name="export", short_help="Commands to export your EveryVoice models"
)

# Add the segment commands
segment_group = typer.Typer(
    **default_typer_args,
    help=CLI_LONG_HELP,
)


command(
    segment_group,
    short_help=ALIGN_SINGLE_SHORT_HELP,
    name="align",
    help=ALIGN_SINGLE_LONG_HELP,
)(ctc_segment_align_single)

command(
    segment_group,
    short_help=EXTRACT_SEGMENTS_SHORT_HELP,
    name="extract",
    help=EXTRACT_SEGMENTS_LONG_HELP,
)(extract_segments_from_textgrid)

app.add_typer(
    segment_group, name="segment", short_help="Commands to align and segment audio"
)


@command(
    app,
    no_args_is_help=False,
    short_help="Create configuration files for a new project",
    help="""
    # This command will help you create all the configuration necessary for using a new project.

    ## Getting Started

    In order to get started, please:

    1. Make sure your audio data is available on your computer and in .wav format.

    2. Have a 'metadata' file that minimally has two columns, one for the text of the audio and one for the basename of the file.

    ## Extra details and examples

    Inside /path/to/wavs/ you should have wav audio files like test0001.wav, test0002.wav etc - they can be called anything you want but the part of the file (minus the .wav portion) must be in your metadata file.

    Example wavs directory: "/path/to/wavs/"

    Example metadata: [https://github.com/EveryVoiceTTS/EveryVoice/blob/main/everyvoice/filelists/lj_full.psv](https://github.com/EveryVoiceTTS/EveryVoice/blob/main/everyvoice/filelists/lj_full.psv)

""",
)
def new_project(
    trace: Annotated[
        bool, typer.Option(help="Enable question tree trace mode.", hidden=True)
    ] = False,
    debug_state: Annotated[
        bool, typer.Option(help="Enable wizard state debug/trace mode.", hidden=True)
    ] = False,
    resume_from: Annotated[
        Optional[Path],
        typer_file_option(
            "--resume-from",
            "-r",
            help="Resume from previously saved progress.",
        ),
    ] = None,
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


preprocess_group = typer.Typer(
    **default_typer_args,
    help="""
    # Preprocess Help

    Preprocess your data before training.

        - **text-to-spec** --- preprocess data for a FastSpeech2 (text-to-spec) model.

        - **text-to-wav** --- preprocess data for a StyleTTS2 (text-to-wav) model.
            Only audio and text steps are run; pitch, energy, and spectrogram
            features are computed on-the-fly during training.
    """,
)


command(
    preprocess_group,
    name="text-to-spec",
    short_help="Preprocess data for text-to-spec (FastSpeech2) training",
    help=f"""Preprocess data for a FastSpeech2 text-to-spec model.

    **everyvoice preprocess text-to-spec config/{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml**

    To run only specific steps:

    **everyvoice preprocess text-to-spec config/{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml -s energy -s pitch**
    """,
)(preprocess_fs2)

command(
    preprocess_group,
    name="spec-to-wav",
    short_help="Preprocess data for spec-to-wav (HiFiGAN) training",
    help=f"""Preprocess data for a HiFiGAN spec-to-wav model.

    **everyvoice preprocess spec-to-wav config/{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml**
    """,
)(preprocess_hfg)

command(
    preprocess_group,
    name="text-to-wav",
    short_help="Preprocess data for text-to-wav (StyleTTS2) training",
    help=f"""Preprocess data for a StyleTTS2 text-to-wav model.

    **everyvoice preprocess text-to-wav config/{TEXT_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml**

    To also preprocess an out-of-distribution (OOD) text file:

    **everyvoice preprocess text-to-wav config/{TEXT_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml --ood-data-file data/ood_texts.txt**
    """,
)(preprocess_styletts2)

app.add_typer(
    preprocess_group, name="preprocess", short_help="Commands to preprocess your data"
)


# Add the train commands
train_group = typer.Typer(
    **default_typer_args,
    help="""
    # Train Help

    TODO: Please visit http://pathtomodelinfodocs for more information

        - **text-to-spec** --- this is the most common model you will need to train. It is a model from text inputs to spectral feature (aka spectrogram) outputs.

        - **spec-to-wav** --- this is the model that turns your spectral features into audio. It is also known as a 'vocoder'. You will typically not need to train your own version. Please refer to [https://pathtocheckpoints](https://pathtocheckpoints) for more information.

        - **text-to-wav** --- this is a model that directly trains text input to waveform/audio output. It is much more heavy duty than training a text-to-spec model. A text-to-spec model will take less than 1 day to train for most datasets, while a text-to-wav model will take over a week for most datasets. For that reason, we strongly recommend training a text-to-wav model first. If you are satisfied with the results, and have extra GPU resources, you can try training a text-to-wav model for improved sentence-level audio quality.

    """,
)

command(
    train_group,
    name="text-to-spec",
    short_help="Train your Text-to-Spec (FastSpeech2) model",
    help=f"""Train your text-to-spec model.  For example:

    **everyvoice train text-to-spec config/{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml**
    """,
)(train_fs2)

command(
    train_group,
    name="spec-to-wav",
    short_help="Train your Spec-to-Wav (HiFiGAN) model",
    help=f"""Train your spec-to-wav model.  For example:

    **everyvoice train spec-to-wav config/{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml**
    """,
)(train_hfg)

command(
    train_group,
    name="text-to-wav",
    short_help="Train an end-to-end (StyleTTS2) model",
    help=f"""Train an end-to-end text-to-speech model. For example:

    **everyvoice train text-to-wav config/{TEXT_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml --mode first**
    """,
)(train_styletts2)

app.add_typer(
    train_group, name="train", short_help="Commands to train your EveryVoice models"
)

# Add synthesize commands
synthesize_group = typer.Typer(
    **default_typer_args,
    help="""
    # Synthesize Help

        - **from-text** --- This is the most common input for performing normal speech synthesis. It will take text or a filelist with text and produce either waveform audio or spectrogram. This option uses FastSpeech2 & HiFiGAN. If you want to do end-to-end synthesis with StyleTTS2, run `everyvoice synthesize text-to-wav` instead.

         - **text-to-wav** --- Synthesize audio directly from text using a trained end-to-end (StyleTTS2) model. Only supports the wav output format.

        - **from-spec** --- This is the model that turns your spectral features into audio. This type of synthesis is also known as copy synthesis and unless you know what you are doing, you probably don't want to do this.

    """,
)

command(synthesize_group, name="from-text")(synthesize_fs2)

command(synthesize_group, name="from-spec")(synthesize_hfg)

command(
    synthesize_group,
    name="text-to-wav",
    short_help="Synthesize audio from text using a trained StyleTTS2 model",
)(synthesize_styletts2)

app.add_typer(
    synthesize_group,
    name="synthesize",
    short_help="Commands to synthesize using your pre-trained EveryVoice models",
)

# Add fetch-pretrained commands
fetch_pretrained_group = typer.Typer(
    **default_typer_args,
    help="""
    # Fetch Pretrained Models

    Download pretrained model weights from HuggingFace before running a training
    job on a cluster node that has no internet access.  Run this command on the
    head node first; the files are cached by the HuggingFace hub and reused
    automatically during training.
    """,
)

command(
    fetch_pretrained_group,
    name="text-to-wav",
    short_help="Download pretrained weights for StyleTTS2 training",
)(fetch_pretrained_styletts2)

app.add_typer(
    fetch_pretrained_group,
    name="fetch-pretrained",
    short_help="Commands to download pretrained model weights from HuggingFace",
)

# check group
check_group = typer.Typer(**default_typer_args)

# Add check_data to root
command(
    check_group,
    name="data",
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


def require_exactly_one_of(arg1: Any, arg1_name: str, arg2: Any, arg2_name: str):
    if arg1 and arg2:
        raise typer.BadParameter(
            f"Please specify only one of {arg1_name} or {arg2_name}."
        )
    if not arg1 and not arg2:
        raise typer.BadParameter(f"One of {arg1_name} and {arg2_name} is required.")


def open_text_or_psv_file(
    text_file: Optional[Path], psv_file: Optional[Path], language: Optional[str]
) -> list[dict[str, str]]:
    """helper for check_text_config: Open a text or psv file into records.

    Language is required if not already in the psv

    raises: typer.BadParameter if something is wrong"""
    from everyvoice.utils import generic_psv_filelist_reader

    if text_file:
        with open(text_file, "r", encoding="utf8") as f:
            text_lines = list(f)
        # print(text_lines)
        if language is None:
            raise typer.BadParameter("--language is required with --text-file.")
        records = [{"characters": line, "language": language} for line in text_lines]
    else:
        assert psv_file
        records = generic_psv_filelist_reader(psv_file)
        if "language" not in records[0]:
            if language is None:
                raise typer.BadParameter(
                    "--language is required for a psv file without a language column."
                )
            for record in records:
                record["language"] = language
    return records


def get_text_config_from_config_or_model(config: Optional[Path], model: Optional[Path]):
    """Helper for check_text_config: load a TextConfig from a config file or model file"""
    from everyvoice.config.text_config import TextConfig
    from everyvoice.utils import spinner

    if config:
        text_config: TextConfig = TextConfig.load_config_from_path(config)
    else:
        assert model
        with spinner("Loading model"):
            try:
                checkpoint = load_checkpoint(model)
            except Exception as e:
                raise typer.BadParameter(
                    f"Model/checkpoint '{model}' does not appear to be valid.\nError from loader: {e}"
                )
        # print("Looking for text config")
        model_config = checkpoint["hyper_parameters"]["config"]
        if "text" in model_config:
            # FS2 models have hyper_parameters.config.text
            text_config = TextConfig(**model_config["text"])
        elif "ev_config" in model_config and "text" in model_config["ev_config"]:
            # StyleTTS2 models have hyper_parameters.config.ev_config.text
            text_config = TextConfig(**model_config["ev_config"]["text"])
        else:
            # Models without text config, e.g., a HiFiGan Vocoder, are not accepted here
            raise typer.BadParameter(
                f"Model/checkpoint '{model}' does not have an embedded text configuration."
            )
    return text_config


@command(check_group, name="text-config")
def check_text_config(
    config: Annotated[
        Optional[Path],
        typer_file_option(
            "--config",
            "-c",
            help="path to text config, i.e., everyvoice-shared-text.yaml",
        ),
    ] = None,
    model: Annotated[
        Optional[Path],
        typer_file_option(
            "--model", "-m", help="path to a model whose text config will be used"
        ),
    ] = None,
    text_file: Annotated[
        Optional[Path],
        typer_file_option(help="path to a plain text file to check"),
    ] = None,
    psv_file: Annotated[
        Optional[Path],
        typer_file_option(help="path to a psv file to check"),
    ] = None,
    language: Annotated[
        Optional[str],
        typer.Option(
            "--language",
            "-l",
            help="language id, required with --text-file, or for a psv file without a language column. "
            + "Declaring the language is always required, because text normalization can be language specific, and g2p is always language specific.",
        ),
    ] = None,
):
    """
    Inspect a text configuration for compatiblity with an input file

    Test processing input_file against the text configuration provided, or the text
    configuration found in model, and report any incompatibilities.

    Required options: one of --config and --model, as well as one of --text-file and --psv-file.
    """
    from everyvoice.utils import spinner

    require_exactly_one_of(config, "--config", model, "--model")
    require_exactly_one_of(text_file, "--text-file", psv_file, "--psv-file")
    records = open_text_or_psv_file(text_file, psv_file, language)

    # Expensive imports are deferred so we fail fast where we can
    with spinner("Loading software"):
        from everyvoice.config.text_config import TextConfig  # noqa F401
        from everyvoice.preprocessor.preprocessor import Preprocessor
        from everyvoice.text.text_processor import TextProcessor
        from everyvoice.text.utils import guess_graphemes_in_text

    text_config = get_text_config_from_config_or_model(config, model)
    # print(text_config)

    text_processor_chars_only = TextProcessor(text_config)
    text_processor_all = TextProcessor(text_config)
    with spinner("Analyzing text"):
        for record in records:
            # print(record)
            # Process just the text to calculate missing characters
            _ = Preprocessor.process_text(
                record,
                text_processor_chars_only,
                specific_text_representation=TargetTrainingTextRepresentationLevel.characters,
            )
            # Process all to also calculate missing phones
            _ = Preprocessor.process_text(record, text_processor_all)

    missing_characters = text_processor_chars_only.missing_symbols
    missing_phones = text_processor_all.missing_symbols - missing_characters
    missing_symbol_groups = list(missing_characters)
    for missing_symbol_group in missing_symbol_groups:
        split_symbols = guess_graphemes_in_text(missing_symbol_group)
        if len(split_symbols) > 1:
            count = missing_characters.pop(missing_symbol_group)
            for symbol in split_symbols:
                missing_characters[symbol] += count
    # print("Missing characters", missing_characters)
    # print("Missing phones", missing_phones)
    if missing_characters:
        print(
            "The following characters are missing from your text config:",
            sorted(missing_characters),
        )
    if missing_phones:
        print(
            "The following phones are missing from your text config:",
            sorted(missing_phones),
        )


app.add_typer(
    check_group, name="check", short_help="Commands to check your data and/or config"
)

# Add the checkpoint commands
checkpoint_group = typer.Typer(**default_typer_args)
command(
    checkpoint_group,
    name="inspect",
    short_help="Extract structural information from a checkpoint",
)(inspect)

command(
    checkpoint_group,
    name="rename-speaker",
    short_help="Rename a speaker in the checkpoint's parameters",
)(rename_speaker)

app.add_typer(
    checkpoint_group,
    name="checkpoint",
    short_help="Commands to inspect and rename speakers in your EveryVoice checkpoints",
)


@command(app, hidden=True, short_help="Inspect a model checkpoint")
def inspect_checkpoint(model_path: Path):
    """
    Inspect a model checkpoint.
    """
    rich_print(
        Panel(
            dedent(
                f"""
                This command has been renamed to `everyvoice checkpoint inspect`.
                Please use `everyvoice checkpoint inspect {model_path}` instead.
                """
            ).strip(),
            title="Inspect Checkpoint",
        )
    )


AllowedDemoOutputFormats = Enum(  # type: ignore
    "AllowedDemoOutputFormats",
    [("all", "all")] + [(i.name, i.value) for i in SynthesizeOutputFormats],
)

_VOCODER_CLASS_NAMES = {"HiFiGAN", "HiFiGANGenerator"}
_FS2_CLASS_NAMES = {"FastSpeech2"}
_STYLETTS2_CLASS_NAMES = {"StyleTTS2Module"}


def _peek_model_class(checkpoint_path: Path) -> str:
    """Load a checkpoint header and return the stored model class name.

    Returns an empty string for legacy checkpoints that predate the model_info field.
    Raises typer.BadParameter if the file cannot be read as a PyTorch checkpoint.
    """
    import torch

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise typer.BadParameter(
            f"Could not read checkpoint '{checkpoint_path}': {e}",
            param_hint="CHECKPOINT",
        )
    return ckpt.get("model_info", {}).get("name", "")


def _load_list_file(path: Optional[Path]) -> list[str]:
    """Read a plain-text word/utterance list from *path*, one entry per line."""
    if path is None:
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _parse_ref_speakers(ref_speaker: list[str]) -> dict[str, Path]:
    """Parse --ref-speaker 'Name=path' values into a display-name → Path mapping."""
    speakers_dict: dict[str, Path] = {}
    for s in ref_speaker:
        if "=" not in s:
            raise typer.BadParameter(
                f"--ref-speaker '{s}' must be in the format 'Display Name=path/to/audio.wav'.",
                param_hint="--ref-speaker",
            )
        display_name, path_str = s.split("=", 1)
        audio_path = Path(path_str.strip()).expanduser()
        if not audio_path.exists():
            raise typer.BadParameter(
                f"Reference audio file not found: {audio_path}",
                param_hint="--ref-speaker",
            )
        speakers_dict[display_name.strip()] = audio_path
    return speakers_dict


def _load_fs2_ui_config(ui_config_file: Optional[Path]) -> "dict | None":
    """Read and JSON-parse the FS2 UI config file, or return None if not given."""
    import json

    if ui_config_file is None:
        print("  - UI Config file: None")
        return None
    print(f"  - UI Config file: {ui_config_file}")
    with open(ui_config_file) as f:
        try:
            ui_config_json = json.load(f)
            print("\t config loaded")
            return ui_config_json
        except Exception as e:
            raise typer.BadParameter(
                f"Your config file {ui_config_file} has errors.\n {e}"
            )


def _run_styletts2_demo(
    checkpoint: Path,
    ref_speaker: list[str],
    reference: Optional[Path],
    speakers: list[str],
    vocoder: Optional[Path],
    allowlist: Optional[Path],
    denylist: Optional[Path],
    allowlist_data: list[str],
    denylist_data: list[str],
    output_dir: Path,
    accelerator: str,
    port: int,
    share: bool,
    server_name: str,
) -> None:
    """Validate StyleTTS2-specific options and launch the Gradio demo."""
    if vocoder is not None:
        raise typer.BadParameter(
            "StyleTTS2 does not use a separate vocoder. Remove --vocoder.",
            param_hint="--vocoder",
        )
    if speakers != ["all"]:
        raise typer.BadParameter(
            "StyleTTS2 does not use --speaker as a filter. "
            "To define named speakers with reference audio use --ref-speaker 'Name=path/to/audio.wav'.",
            param_hint="--speaker",
        )
    if not ref_speaker and reference is None:
        raise typer.BadParameter(
            "Provide at least one --ref-speaker 'Name=path/to/audio.wav' or a --reference path.",
            param_hint="--ref-speaker / --reference",
        )

    speakers_dict = _parse_ref_speakers(ref_speaker)
    default_reference = reference if not speakers_dict else None

    import json
    import tempfile

    import torch

    print("INFO - Starting the StyleTTS2 demo with the following parameters:")
    print(f"  - Checkpoint:     {checkpoint}")
    try:
        _state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _hp = _state.get("hyper_parameters", {})
        if "config" in _hp:
            print("  - Checkpoint config:")
            print(json.dumps(_hp["config"], indent=4, default=str))
        del _state
    except Exception as e:
        print(f"  - (Could not read checkpoint config: {e})")
    if speakers_dict:
        for name, path in speakers_dict.items():
            print(f"  - Ref speaker:    {name} = {path}")
    else:
        print(f"  - Reference:      {reference}")
    print(f"  - Allowlist:      {allowlist if allowlist else 'None'}")
    print(f"  - Denylist:       {denylist if denylist else 'None'}")
    print(f"  - Output Dir:     {output_dir}")
    print(f"  - Accelerator:    {accelerator}")
    print(f"  - Port:           {port}")
    print(f"  - Share:          {share}")
    print(f"  - Server Name:    {server_name}")

    from everyvoice.utils import spinner

    with spinner("Loading software"):
        from everyvoice.demo.app import create_demo_app_styletts2

    with spinner("Loading model"):
        demo_app = create_demo_app_styletts2(
            model_path=checkpoint,
            output_dir=output_dir,
            speakers=speakers_dict,
            default_reference=default_reference,
            accelerator=accelerator,
            allowlist=allowlist_data,
            denylist=denylist_data,
        )

    demo_app.launch(
        share=share,
        server_port=port,
        server_name=server_name,
        allowed_paths=[str(output_dir), tempfile.gettempdir()],
    )


def _run_fs2_demo(
    checkpoint: Path,
    vocoder: Optional[Path],
    speakers: list[str],
    languages: list[str],
    outputs: list,
    ui_config_file: Optional[Path],
    ref_speaker: list[str],
    reference: Optional[Path],
    allowlist: Optional[Path],
    denylist: Optional[Path],
    allowlist_data: list[str],
    denylist_data: list[str],
    output_dir: Path,
    accelerator: str,
    port: int,
    share: bool,
    server_name: str,
    **kwargs,
) -> None:
    """Validate FastSpeech2-specific options and launch the Gradio demo."""
    if vocoder is None:
        raise typer.BadParameter(
            "FastSpeech2 requires a vocoder checkpoint. "
            "Pass it with --vocoder path/to/hifigan.ckpt.",
            param_hint="--vocoder",
        )
    if ref_speaker:
        raise typer.BadParameter(
            "--ref-speaker is only used with StyleTTS2 models. "
            "To filter FastSpeech2 speakers use --speaker.",
            param_hint="--ref-speaker",
        )
    if reference is not None:
        raise typer.BadParameter(
            "--reference is only used with StyleTTS2 models.",
            param_hint="--reference",
        )

    print("INFO - Starting the demo with the following parameters:")
    print(f"  - Checkpoint:     {checkpoint}")
    print(f"  - Vocoder:        {vocoder}")
    print(f"  - Languages:      {languages}")
    print(f"  - Speakers:       {speakers}")
    print(f"  - Outputs:        {outputs}")
    print(f"  - Output Dir:     {output_dir}")
    print(f"  - Accelerator:    {accelerator}")
    print(f"  - Allowlist:      {allowlist_data if allowlist else 'None'}")
    print(f"  - Denylist:       {denylist_data if denylist else 'None'}")
    print(f"  - Port:           {port}")
    print(f"  - Share:          {share}")
    print(f"  - Server Name:    {server_name}")
    ui_config_json = _load_fs2_ui_config(ui_config_file)

    from everyvoice.utils import spinner

    with spinner("Loading software"):
        import tempfile

        from everyvoice.demo.app import create_demo_app

    with spinner("Loading models"):
        demo_app = create_demo_app(
            text_to_spec_model_path=checkpoint,
            spec_to_wav_model_path=vocoder,
            languages=languages,
            speakers=speakers,
            outputs=outputs,
            output_dir=output_dir,
            accelerator=accelerator,
            allowlist=allowlist_data,
            denylist=denylist_data,
            app_ui_config=ui_config_json,
            **kwargs,
        )

    demo_app.launch(
        share=share,
        server_port=port,
        server_name=server_name,
        allowed_paths=[str(output_dir), tempfile.gettempdir()],
    )


@command(
    app,
    name="demo",
    short_help="Launch an interactive Gradio demo for any EveryVoice model",
)
@merge_args(inference_base_command_interface)
def demo(
    checkpoint: Annotated[
        Path,
        typer_file_argument(
            help="Path to a trained EveryVoice checkpoint (.ckpt). "
            "The model type is detected automatically from the checkpoint. "
            "For FastSpeech2 models also pass --vocoder."
        ),
    ],
    # ---- FastSpeech2 options ------------------------------------------------
    vocoder: Annotated[
        Optional[Path],
        typer_file_option(
            "--vocoder",
            "-V",
            help="[FastSpeech2] Path to a trained HiFiGAN vocoder checkpoint. "
            "Required when the primary checkpoint is a FastSpeech2 model. "
            "Not used with StyleTTS2 checkpoints.",
            rich_help_panel="FastSpeech2 (text-to-spec) Options",
        ),
    ] = None,
    speakers: list[str] = typer.Option(
        ["all"],
        "--speaker",
        "-s",
        help="[FastSpeech2] Speaker names to expose in the demo UI. "
        "Repeat the flag to include multiple speakers. "
        "Defaults to all speakers in the model. "
        "Not applicable to StyleTTS2 — use --ref-speaker instead.",
        rich_help_panel="FastSpeech2 (text-to-spec) Options",
    ),
    languages: list[str] = typer.Option(
        ["all"],
        "--language",
        "-l",
        help="[FastSpeech2] Languages to expose in the demo UI. "
        "Repeat the flag to include multiple languages. "
        "Defaults to all languages in the model.",
        rich_help_panel="FastSpeech2 (text-to-spec) Options",
    ),
    outputs: list[AllowedDemoOutputFormats] = typer.Option(
        ["all"],
        "--output-format",
        "-O",
        help="[FastSpeech2] Output formats to expose in the demo UI.",
        rich_help_panel="FastSpeech2 (text-to-spec) Options",
    ),
    ui_config_file: Annotated[
        Optional[Path],
        typer_file_option(
            "--ui-config-file",
            "-C",
            help="[FastSpeech2] JSON file to override UI labels (app_title, app_description, "
            "app_instructions, speakers, languages, input_text_label, "
            "duration_multiplier_label, language_label, speaker_label, "
            "output_format_label, synthesize_label, file_output_label).",
            rich_help_panel="FastSpeech2 (text-to-spec) Options",
        ),
    ] = None,
    # ---- StyleTTS2 options --------------------------------------------------
    ref_speaker: list[str] = typer.Option(
        [],
        "--ref-speaker",
        "-R",
        help="[StyleTTS2] Named speaker with reference audio, in the format "
        "'Display Name=path/to/audio.wav'. "
        "Repeat the flag to add multiple speakers. "
        "Their style encodings are pre-computed at startup and shown in a dropdown. "
        "Example: --ref-speaker 'Eric=eric.wav' --ref-speaker 'Darlene=darlene.wav'",
        rich_help_panel="StyleTTS2 (text-to-wav) Options",
    ),
    reference: Optional[Path] = typer.Option(
        None,
        "--reference",
        "-r",
        help="[StyleTTS2] Default reference audio file that sets the initial speaker style. "
        "Use this for reference-upload mode (no speaker dropdown). "
        "Use --ref-speaker instead to pre-define named speakers.",
        exists=True,
        rich_help_panel="StyleTTS2 (text-to-wav) Options",
    ),
    # ---- Shared options -----------------------------------------------------
    allowlist: Annotated[
        Optional[Path],
        typer_file_option(
            "--allowlist",
            help="Plain text file with allowed words or utterances (one per line). "
            "All other input is rejected. Cannot be combined with --denylist.",
        ),
    ] = None,
    denylist: Annotated[
        Optional[Path],
        typer_file_option(
            "--denylist",
            help="Plain text file with disallowed words or utterances (one per line). "
            "All other input is allowed. Cannot be combined with --allowlist. "
            "IMPORTANT: there are many ways to bypass a denylist. "
            "Use --allowlist for maximum security.",
        ),
    ] = None,
    output_dir: Path = typer_directory_option(
        "synthesis_output",
        "--output-dir",
        "-o",
        exists=False,
        help="Directory where synthesized audio files are written.",
    ),
    accelerator: str = typer.Option(
        "auto",
        "--accelerator",
        "-a",
        help="PyTorch Lightning accelerator (e.g. 'auto', 'cpu', 'gpu').",
    ),
    port: int = typer.Option(7860, "--port", "-p", help="Port to serve the demo on."),
    share: bool = typer.Option(
        False,
        "--share",
        help="Publish the demo via Gradio's share tunnel (accessible from the internet).",
    ),
    server_name: str = typer.Option(
        "0.0.0.0",
        "--server-name",
        "-n",
        help="Host/IP address to bind the demo server to.",
    ),
    **kwargs,
):
    """Launch an interactive Gradio demo for any EveryVoice model.

    The model type is detected automatically from the checkpoint.
    Pass a single checkpoint for **StyleTTS2** (text-to-wav) models:

        everyvoice demo path/to/styletts2.ckpt --ref-speaker 'Eric=eric.wav'

    Pass a FastSpeech2 (text-to-spec) checkpoint plus a vocoder (spec-to-wav) for **FastSpeech2 + HiFiGAN** models:

        everyvoice demo path/to/fs2.ckpt --vocoder path/to/hifigan.ckpt
    """
    if allowlist and denylist:
        raise typer.BadParameter(
            "Provide either --allowlist or --denylist, not both.",
        )

    model_class = _peek_model_class(checkpoint)

    if model_class in _VOCODER_CLASS_NAMES:
        raise typer.BadParameter(
            f"'{checkpoint}' appears to be a standalone vocoder checkpoint ({model_class}). "
            "Pass your FastSpeech2 checkpoint as the CHECKPOINT argument and provide "
            "this vocoder with --vocoder.",
            param_hint="CHECKPOINT",
        )
    if model_class not in _FS2_CLASS_NAMES | _STYLETTS2_CLASS_NAMES:
        raise typer.BadParameter(
            f"Unrecognized model type '{model_class}' in checkpoint '{checkpoint}'. "
            "Expected a FastSpeech2 or StyleTTS2 checkpoint.",
            param_hint="CHECKPOINT",
        )

    allowlist_data = _load_list_file(allowlist)
    denylist_data = _load_list_file(denylist)

    shared = dict(
        allowlist=allowlist,
        denylist=denylist,
        allowlist_data=allowlist_data,
        denylist_data=denylist_data,
        output_dir=output_dir,
        accelerator=accelerator,
        port=port,
        share=share,
        server_name=server_name,
    )

    if model_class in _STYLETTS2_CLASS_NAMES:
        _run_styletts2_demo(
            checkpoint, ref_speaker, reference, speakers, vocoder, **shared  # type: ignore[arg-type]
        )
    else:
        _run_fs2_demo(
            checkpoint,
            vocoder,
            speakers,
            languages,
            outputs,
            ui_config_file,
            ref_speaker,
            reference,
            **shared,  # type: ignore[arg-type]
            **kwargs,
        )


# Deferred full initialization to optimize the CLI, but still exposed for unit testing.
SCHEMAS_TO_OUTPUT: dict[str, Any] = {}  # dict[str, type[BaseModel]]


@command(app, no_args_is_help=False, hidden=True)
def update_schemas(
    out_dir: Annotated[
        Optional[Path],
        typer_directory_option("-o", "--out-dir"),
    ] = None,
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
    import json

    from everyvoice.config.preprocessing_config import PreprocessingConfig
    from everyvoice.config.text_config import TextConfig
    from everyvoice.model.e2e.config import E2EConfig
    from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
    from everyvoice.model.vocoder.config import VocoderConfig

    # We should not be changing the schema for patches, so only include major/minor version
    MAJOR_MINOR_VERSION = ".".join(VERSION.split(".")[:2])

    SCHEMAS_TO_OUTPUT.update(
        {
            f"{TEXT_TO_WAV_CONFIG_FILENAME_PREFIX}-{MAJOR_MINOR_VERSION}.json": E2EConfig,
            f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}-{MAJOR_MINOR_VERSION}.json": FeaturePredictionConfig,
            f"{PREPROCESSING_CONFIG_FILENAME_PREFIX}-{MAJOR_MINOR_VERSION}.json": PreprocessingConfig,
            f"{TEXT_CONFIG_FILENAME_PREFIX}-{MAJOR_MINOR_VERSION}.json": TextConfig,
            f"{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}-{MAJOR_MINOR_VERSION}.json": VocoderConfig,
        }
    )

    all_good = True
    for filename, schema in SCHEMAS_TO_OUTPUT.items():
        schema_contents = json.dumps(schema.model_json_schema(), indent=2) + "\n"
        if (schema_dir_path / filename).exists():
            with open(schema_dir_path / filename) as f:
                existing_contents = f.read()
                if existing_contents == schema_contents:
                    print(f"Schema '{filename}' already up to date.")
                else:
                    all_good = False
                    print(
                        f"Out of date schema '{filename}' exists in '{schema_dir_path}'."
                    )
        else:
            with open(
                schema_dir_path / filename, "w", encoding="utf8", newline="\n"
            ) as f:
                f.write(schema_contents)
                print(f"Schema '{filename}' created.")

    if not all_good:
        sys.exit(
            dedent(
                """
                ERROR: out-of-date schemas exist.
                If the current schemas were already published to the schema store, please
                bump the EveryVoice minor version number and run update-schemas again.
                If the current minor version is still in development, delete the out-of-date
                schemas and try again."""
            )
        )


@command(app)
def g2p(
    lang_id: Annotated[str, typer.Argument(help="language id")],
    # Ignoring mypy since class FileText(io.TextIOWrapper)
    input_file: Annotated[typer.FileText, typer.Argument()] = TextIOWrapper(
        sys.stdin.buffer,
        encoding="utf-8",
    ),  # type: ignore[assignment]
    config: Annotated[
        Optional[Path], typer.Option(help="path to everyvoice-shared-text.yaml")
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


if __name__ == "__main__":
    app()
