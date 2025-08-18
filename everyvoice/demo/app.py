import json
import os
import string
import subprocess
import sys
from functools import partial
from unicodedata import normalize

import gradio as gr
import torch
from loguru import logger

from everyvoice.cli import AllowedDemoOutputFormats
from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.synthesize import (
    synthesize_helper,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
    FastSpeech2,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions import (
    SynthesizeOutputFormats,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.utils import (
    truncate_basename,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
    HiFiGAN,
    HiFiGANConfig,
    HiFiGANGenerator,
    load_hifigan_from_checkpoint,
)
from everyvoice.utils import slugify
from everyvoice.utils.heavy import get_device_from_accelerator

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
# list[str] would also be allowed, but we want to keep it a list of tuples to allow for decoupling model internal slugs and display names.
GradioChoices = list[
    tuple[str, str]
]  # expected format: [(display-name, form-value), ...]


def synthesize_audio(
    text,
    duration_control,
    language,
    speaker,
    output_format,
    style_reference,
    text_to_spec_model,
    vocoder_model,
    vocoder_config,
    accelerator,
    device,
    allowlist,
    denylist,
    output_dir=None,
    include_file_output=True,
):
    if text == "":
        raise gr.Error(
            "Text for synthesis was not provided. Please type the text you want to be synthesized into the textfield."
        )
    norm_text = normalize_text(text)
    basename = truncate_basename(slugify(text))
    if allowlist and norm_text not in allowlist:
        raise gr.Error(
            f"Oops, the word {text} is not allowed to be synthesized by this model. Please contact the model owner."
        )
    if denylist:
        norm_words = norm_text.split()
        for word in norm_words:
            if word in denylist:
                raise gr.Error(
                    f"Oops, the word {text} is not allowed to be synthesized by this model. Please contact the model owner."
                )
    if language is None:
        raise gr.Error("Language is not selected. Please select a language.")
    if speaker is None:
        raise gr.Error("Speaker is not selected. Please select a speaker.")
    if output_format is None:
        raise gr.Error("Output format is not selected. Please select an output format.")
    config, device, predictions, callbacks = synthesize_helper(
        model=text_to_spec_model,
        style_reference=style_reference,
        vocoder_model=vocoder_model,
        vocoder_config=vocoder_config,
        texts=[text],
        language=language,
        accelerator=accelerator,
        devices="1",
        device=device,
        global_step=text_to_spec_model.config.training.max_steps,
        vocoder_global_step=vocoder_model.config.training.max_steps,
        output_type=(output_format, SynthesizeOutputFormats.wav),
        text_representation=TargetTrainingTextRepresentationLevel.characters,
        output_dir=output_dir,
        speaker=speaker,
        duration_control=duration_control,
        filelist=None,
        filelist_data=None,
        teacher_forcing_directory=None,
        batch_size=16,
        num_workers=1,
    )

    wav_writer = callbacks[SynthesizeOutputFormats.wav]
    wav_output = wav_writer.last_file_written

    file_output = None
    if output_format != SynthesizeOutputFormats.wav:
        file_writer = callbacks[output_format]
        file_output = file_writer.get_filename(basename, speaker, language)

    if include_file_output:
        return wav_output, file_output
    else:
        return wav_output


def require_ffmpeg():
    """Make sure ffmpeg is found and can be run, or else exit"""
    try:
        subprocess.run(["ffmpeg", "-h"], capture_output=True)
    except Exception:
        logger.error(
            "ffmpeg not found or cannot be executed.\nffmpeg is required to run the demo.\nPlease install it, e.g., with 'conda install ffmpeg' or with your OS's package manager."
        )
        sys.exit(1)


def normalize_text(text: str) -> str:
    """Normalize text in allowlist and denylist to prevent hacking.
            this is extremely deficient in its current state and only
            prevents:

            - Unicode homograph attacks
            - Case attacks
            - extraneous punctuation

        Args:
            text (str): an un-normalized word or utterance

        Returns:
            str: normalized text

    >>> normalize_text('FoOBar')
    'fobar'

    >>> normalize_text('fo\u0301obar')
    'f\u00F3obar'

    >>> normalize_text('foobar.')
    'fobar'

    >>> normalize_text('ffoobbaaarr')
    'fobar'

    """
    # Unicode normalization
    text = normalize("NFC", text)

    # Case normalization
    text = text.lower()

    # Remove Punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove duplicate characters
    text = "".join(dict.fromkeys(text))

    return text


def set_speaker_list(speakers: list[str], model_speakers: list[str]) -> GradioChoices:
    speaker_list: GradioChoices = []
    if speakers == ["all"]:
        speaker_list = [(speaker, speaker) for speaker in model_speakers]
    else:
        for speaker in speakers:
            if speaker in model_speakers:
                speaker_list.append((speaker, speaker))
            else:
                print(
                    f"Attention: The model have not been trained for speech synthesis with '{speaker}' speaker. The '{speaker}' speaker option will not be available for selection."
                )
    if speaker_list == []:
        raise ValueError(
            f"Speaker option has been activated, but valid speakers have not been provided. The model has been trained with {model_speakers} speakers. Please select either 'all' or at least some of them."
        )
    return speaker_list


def set_language_list(
    languages: list[str], model_languages: list[str]
) -> GradioChoices:
    language_list: GradioChoices = []
    if languages == ["all"]:
        language_list = [(language, language) for language in model_languages]
    else:
        for language in languages:
            if language in model_languages:
                language_list.append((language, language))
            else:
                print(
                    f"Attention: The model have not been trained for speech synthesis in '{language}' language. The '{language}' language option will not be available for selection."
                )
    if language_list == []:
        raise ValueError(
            f"Language option has been activated, but valid languages have not been provided. The model has been trained in {model_languages} languages. Please select either 'all' or at least some of them."
        )

    return language_list


def load_app_ui_labels(
    ui_config_json_path: os.PathLike | None = None,
    speakers: list[str] = ["all"],
    languages: list[str] = ["all"],
    model_languages: list[str] = [],
    model_speakers: list[str] = [],
) -> tuple[GradioChoices, GradioChoices, dict[str, str]]:
    """Load the app config JSON file if provided and update the speaker and language lists.
    This function checks if the provided ui_config_json_path is a valid JSON file and contains
    the 'speakers' and 'languages' keys. If they are present, it updates the speaker and language lists accordingly.
    Args:
        ui_config_json_path (os.PathLike | None): Path to the app config JSON file.
        speakers (list[str]): List of speakers to be used in the app.
        languages (list[str]): List of languages to be used in the app.
        speak_list (list[str]): List of speakers from the model.
        lang_list (list[str]): List of languages from the model.
    Returns:
        tuple: A tuple of [display-name, form-value] for speakers, languages and the app config json.
    Raises:
        ValueError: If the 'speakers' or 'languages' are not checkpoint or keys in the app config JSON do not match the provided speakers or languages, or if the keys are not dictionaries.
    """
    # tuple of (display-name, form-value) for gradio dropdown
    language_list: GradioChoices = set_language_list(languages, model_languages)
    speaker_list: GradioChoices = set_speaker_list(speakers, model_speakers)
    app_ui_labels = {}  # dict[str,str]  # app UI config JSON

    # json config file is passed
    if ui_config_json_path and str(ui_config_json_path).lower().endswith(".json"):

        # Load the app config JSON file if provided
        with open(ui_config_json_path, "r") as f:
            app_ui_config = json.load(f)  # type: dict[str, dict[str, str] | str]
        # Update the app config with the current settings
        if "speakers" in app_ui_config:
            if not isinstance(app_ui_config["speakers"], dict):
                raise ValueError(
                    "The 'speakers' key in the app config JSON must be a dictionary."
                )
            if ":".join(app_ui_config["speakers"].keys()) != ":".join(
                [row[0] for row in speaker_list]
            ):

                raise ValueError(
                    "The 'speakers' key in the app config JSON does not match the speakers provided."
                )

            speaker_list.clear()
            if speakers == ["all"]:
                # if all speakers are selected, use the speakers from the app config JSON
                speaker_list.extend(
                    [
                        (str(app_ui_config["speakers"].get(speaker, speaker)), speaker)
                        for speaker in app_ui_config["speakers"]
                    ]
                )
            else:
                for speaker in speakers:
                    if speaker in app_ui_config["speakers"]:
                        speaker_list.append(
                            (
                                str(app_ui_config["speakers"].get(speaker, speaker)),
                                speaker,
                            )
                        )

            print("\n\tUsing speakers from app config JSON:", speaker_list)
        if "languages" in app_ui_config:
            if not isinstance(app_ui_config["languages"], dict):
                raise ValueError(
                    "The 'languages' key in the app config JSON must be a dictionary."
                )

            if ":".join(app_ui_config["languages"].keys()) != ":".join(
                [row[0] for row in language_list]
            ):
                raise ValueError(
                    "The 'languages' key in the app config JSON does not match the languages provided."
                )

            # apply language list contraints

            language_list.clear()
            if languages == ["all"]:
                # if all languages are selected, use the languages from the app config JSON
                language_list.extend(
                    [
                        (
                            str(app_ui_config["languages"].get(language, language)),
                            language,
                        )
                        for language in app_ui_config["languages"]
                    ]
                )
            else:
                # if specific languages are selected, use only those from the app config JSON
                for language in languages:
                    if language in app_ui_config["languages"]:
                        language_list.append(
                            (
                                str(app_ui_config["languages"].get(language, language)),
                                language,
                            )
                        )
            print("\n\tUsing languages from app config JSON:", language_list)
        if "app_title" in app_ui_config:
            print(
                f"\n\tUsing app title from app config JSON: {app_ui_config['app_title']}"
            )
        for key in app_ui_config:
            if key not in ["speakers", "languages"]:
                app_ui_labels[key] = str(app_ui_config[key])

    return speaker_list, language_list, app_ui_labels


def make_gradio_display(
    language_list: GradioChoices,
    speaker_list: GradioChoices,
    outputs: list,  # list[str | AllowedDemoOutputFormats]
    output_list: list,
    model: FastSpeech2,
    synthesize_audio_preset: partial,
    app_ui_config: dict[str, str],
) -> gr.Blocks:
    """Create the Gradio Blocks for the demo app."""

    default_language = language_list[0][1]
    interactive_language = len(language_list) > 1
    default_speaker = speaker_list[0][1]
    interactive_speaker = len(speaker_list) > 1
    default_output = output_list[0]
    interactive_output = len(output_list) > 1
    app_title = app_ui_config.get("app_title", "EveryVoice Demo App")
    app_description = app_ui_config.get("app_description", "")
    app_instructions = app_ui_config.get("app_instructions", "")
    helper_text = ""

    if app_description:
        helper_text += f"<p align='center'>{app_description}</p>"
        print(f"\n\tUsing app description from app config JSON: {app_description}")
    if app_instructions:
        helper_text += f"<h5 style='color:#777;margin-bottom:0;padding-bottom:0;'>How to use this app</h5><p style='color:#999;margin-top:0.1em;padding-top:0;'>{app_instructions}</p>"
        print(f"\n\tUsing app instructions from app config JSON: {app_instructions}")
    with gr.Blocks() as demo:
        gr.Markdown(
            f"""
            <h1 align="center">{app_title}</h1>
            {helper_text}
            """
        )
        with gr.Row():
            with gr.Column():
                inp_text = gr.Text(
                    placeholder="This text will be turned into speech.",
                    label=app_ui_config.get("input_text_label", "Input Text"),
                )
                inp_slider = gr.Slider(
                    0.75,
                    1.75,
                    1.0,
                    step=0.25,
                    label=app_ui_config.get(
                        "duration_multiplier_label", "Duration Multiplier"
                    ),
                )
                with gr.Row():
                    inp_lang = gr.Dropdown(
                        choices=language_list,
                        value=default_language,
                        interactive=interactive_language,
                        label=app_ui_config.get("language_label", "Language"),
                    )
                    inp_speak = gr.Dropdown(
                        choices=speaker_list,
                        value=default_speaker,
                        interactive=interactive_speaker,
                        label=app_ui_config.get("speaker_label", "Speaker"),
                    )
                inputs = [inp_text, inp_slider, inp_lang, inp_speak]
                if output_list != [SynthesizeOutputFormats.wav]:
                    with gr.Row():
                        output_format = gr.Dropdown(
                            choices=output_list,
                            value=default_output,
                            interactive=interactive_output,
                            label=app_ui_config.get(
                                "output_format_label", "Output Format"
                            ),
                        )
                    inputs.append(output_format)
                else:
                    synthesize_audio_preset = partial(
                        synthesize_audio_preset,
                        output_format=SynthesizeOutputFormats.wav,
                        include_file_output=False,
                    )
                if model.config.model.use_global_style_token_module:
                    with gr.Row():
                        style_reference = gr.Audio(type="filepath")
                btn = gr.Button(app_ui_config.get("synthesize_label", "Synthesize"))
            with gr.Column():
                out_audio = gr.Audio(format="wav")
                if output_list == [SynthesizeOutputFormats.wav]:
                    # When the only output option is wav, don't show the File Output box
                    outputs = [out_audio]
                else:
                    out_file = gr.File(
                        label=app_ui_config.get("file_output_label", "File Output"),
                        elem_id="file_output",
                    )
                    outputs = [out_audio, out_file]
        # Only include the style reference input if the model supports it
        if model.config.model.use_global_style_token_module:
            inputs.append(style_reference)  # type: ignore
        else:
            synthesize_audio_preset = partial(
                synthesize_audio_preset, style_reference=None
            )
        btn.click(
            synthesize_audio_preset,
            inputs=inputs,
            outputs=outputs,
        )
    return demo


def load_model_from_checkpoint(
    text_to_spec_model_path: os.PathLike,
    spec_to_wav_model_path: os.PathLike,
    accelerator: str,
) -> tuple[FastSpeech2, HiFiGAN | HiFiGANGenerator, HiFiGANConfig, torch.device]:
    """Load the text-to-speech model and vocoder from their respective checkpoints."""
    require_ffmpeg()
    device = get_device_from_accelerator(accelerator)
    vocoder_ckpt = torch.load(
        spec_to_wav_model_path, map_location=device, weights_only=True
    )
    vocoder_model, vocoder_config = load_hifigan_from_checkpoint(vocoder_ckpt, device)
    model: FastSpeech2 = FastSpeech2.load_from_checkpoint(text_to_spec_model_path).to(  # type: ignore
        device
    )
    model.eval()
    return model, vocoder_model, vocoder_config, device


def create_demo_app(
    text_to_spec_model_path: os.PathLike,
    spec_to_wav_model_path: os.PathLike,
    languages: list[str],
    speakers: list[str],
    outputs: list,  # list[str | AllowedDemoOutputFormats]
    output_dir: os.PathLike,
    accelerator: str,
    allowlist: list[str] = [],
    denylist: list[str] = [],
    ui_config_json_path: os.PathLike | None = None,
    **kwargs,
) -> gr.Blocks:
    # Early argument validation where possible
    possible_outputs = [x.value for x in SynthesizeOutputFormats]

    if outputs == [AllowedDemoOutputFormats["all"].value] or outputs == [
        AllowedDemoOutputFormats["all"]
    ]:
        output_list = possible_outputs
    else:
        if not outputs:
            raise ValueError(
                f"Empty outputs list. Please specify ['all'] or one or more of {possible_outputs}"
            )
        output_list = []
        for output in outputs:
            value = getattr(output, "value", output)  # Enum->value as str / str->str
            if value not in possible_outputs:
                raise ValueError(
                    f"Unknown output format '{value}'. Valid outputs values are ['all'] or one or more of {possible_outputs}"
                )
            output_list.append(value)

    model, vocoder_model, vocoder_config, device = load_model_from_checkpoint(
        text_to_spec_model_path, spec_to_wav_model_path, accelerator
    )

    from everyvoice.base_cli.helpers import inference_base_command

    inference_base_command(model, **kwargs)

    # normalize allowlist and denylist
    allowlist = [normalize_text(w) for w in allowlist]
    denylist = [normalize_text(w) for w in denylist]
    synthesize_audio_preset = partial(
        synthesize_audio,
        text_to_spec_model=model,
        vocoder_model=vocoder_model,
        vocoder_config=vocoder_config,
        output_dir=output_dir,
        accelerator=accelerator,
        device=device,
        allowlist=allowlist,
        denylist=denylist,
    )
    model_languages = list(model.lang2id.keys())
    model_speakers = list(model.speaker2id.keys())
    speaker_list, language_list, app_ui_config = load_app_ui_labels(
        ui_config_json_path=ui_config_json_path,
        speakers=speakers,
        languages=languages,
        model_languages=model_languages,
        model_speakers=model_speakers,
    )
    return make_gradio_display(
        language_list,
        speaker_list,
        outputs,
        output_list,
        model,
        synthesize_audio_preset,
        app_ui_config,
    )
