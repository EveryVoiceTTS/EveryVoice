import os
import string
import subprocess
import sys
from functools import partial
from pathlib import Path
from unicodedata import normalize

import gradio as gr
import torch
import typer
from loguru import logger

from everyvoice.cli import AllowedDemoOutputFormats
from everyvoice.config.type_definitions import DatasetTextRepresentation
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
    output_dir: Path,
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
        text_representation=DatasetTextRepresentation.characters,
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
        raise typer.BadParameter(
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
        raise typer.BadParameter(
            f"Language option has been activated, but valid languages have not been provided. The model has been trained in {model_languages} languages. Please select either 'all' or at least some of them."
        )

    return language_list


def load_app_ui_labels(
    app_ui_config: dict | None = None,
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
    if app_ui_config is not None:

        # Update the app config with the current settings
        if "speakers" in app_ui_config:
            if not isinstance(app_ui_config["speakers"], dict):
                raise typer.BadParameter(
                    "The 'speakers' key in the app config JSON must be a dictionary."
                )
            if ":".join(app_ui_config["speakers"].keys()) != ":".join(
                [row[0] for row in speaker_list]
            ):

                raise typer.BadParameter(
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
                raise typer.BadParameter(
                    "The 'languages' key in the app config JSON must be a dictionary."
                )

            if ":".join(app_ui_config["languages"].keys()) != ":".join(
                [row[0] for row in language_list]
            ):
                raise typer.BadParameter(
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
    try:
        vocoder_ckpt = torch.load(
            spec_to_wav_model_path, map_location=device, weights_only=True
        )
    except Exception as e:
        raise ValueError(
            f"Error loading HiFiGAN model '{spec_to_wav_model_path}'. It does not appear to be a valid checkpoint."
        ) from e
    try:
        vocoder_model, vocoder_config = load_hifigan_from_checkpoint(
            vocoder_ckpt, device
        )
    except TypeError as e:
        raise ValueError(
            f"Error loading HiFiGAN model '{spec_to_wav_model_path}'. Possible causes: maybe it's not actually a HiFiGAN model? maybe it was trained with an incompatible version of EveryVoice?"
        ) from e
    try:
        model: FastSpeech2 = FastSpeech2.load_from_checkpoint(text_to_spec_model_path).to(  # type: ignore
            device
        )
    except TypeError as e:
        raise ValueError(
            f"Error loading FastSpeech2 model '{text_to_spec_model_path}'. Possible causes: maybe it's not actually an fs2 model? maybe it was trained with an older or a new version of EveryVoice?"
        ) from e

    model.eval()
    return model, vocoder_model, vocoder_config, device


def synthesize_audio_styletts2(
    text: str,
    speaker: "str | None",  # selected display name from dropdown, or None in reference-upload mode
    user_reference,  # filepath from Gradio Audio component, or None if not uploaded / cleared
    diffusion_steps: int,
    embedding_scale: float,
    acoustic_blend: float,
    prosody_blend: float,
    *,
    module,
    mel_transform,
    device,
    output_dir: Path,
    speaker_ref_s: "dict[str, torch.Tensor]",  # pre-computed at startup; empty in reference-only mode
    default_ref_s: "torch.Tensor | None",  # pre-computed from --reference; None in speaker mode
    allowlist: list[str],
    denylist: list[str],
) -> str:
    """Synthesize one utterance with StyleTTS2 and return the path to the saved WAV."""
    if not text or not text.strip():
        raise gr.Error("Please provide text to synthesize.")

    norm_text = normalize_text(text)
    if allowlist and norm_text not in allowlist:
        raise gr.Error(
            f"The text '{text}' is not allowed to be synthesized by this model. "
            "Please contact the model owner."
        )
    if denylist:
        for word in norm_text.split():
            if word in denylist:
                raise gr.Error(
                    f"The text '{text}' contains a word that is not allowed. "
                    "Please contact the model owner."
                )

    # Determine ref_s — prefer user-uploaded audio, then pre-loaded speaker, then default
    if user_reference is not None:
        from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.cli.synthesize import (
            load_reference_style,
        )

        try:
            ref_s = load_reference_style(
                module, mel_transform, Path(user_reference), device
            )
        except Exception as e:
            raise gr.Error(f"Could not load reference audio: {e}")
    elif speaker is not None and speaker in speaker_ref_s:
        ref_s = speaker_ref_s[speaker]
    elif default_ref_s is not None:
        ref_s = default_ref_s
    else:
        raise gr.Error(
            "No reference audio available. Please upload a reference audio file."
        )

    from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.text_utils import (
        TextCleaner,
    )

    text_cleaner = TextCleaner()
    tokens = torch.LongTensor(text_cleaner(text)).unsqueeze(0).to(device)
    if tokens.numel() == 0:
        raise gr.Error(f"Text produced no tokens: {text!r}")
    input_lengths = torch.LongTensor([tokens.size(1)]).to(device)

    try:
        audio = module._synthesize_text(
            tokens,
            input_lengths,
            ref_s=ref_s,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
            acoustic_blend=acoustic_blend,
            prosody_blend=prosody_blend,
        )
    except Exception as e:
        raise gr.Error(str(e))

    import soundfile as sf

    out_path = output_dir / (slugify(text[:50]) + ".wav")
    sf.write(str(out_path), audio, module.sr)
    return str(out_path)


def make_gradio_display_styletts2(
    synthesize_fn,
    speaker_list: "GradioChoices",
    default_reference: "Path | None" = None,
) -> "gr.Blocks":
    """Build the Gradio Blocks for the StyleTTS2 demo.

    When ``speaker_list`` is non-empty a speaker dropdown is shown and the
    reference audio widget becomes an optional style override.  When it is
    empty the reference audio widget is the primary input (reference-upload
    mode) and the ``speaker`` argument is pre-bound as ``None``.
    """
    has_speakers = bool(speaker_list)
    interactive_speaker = len(speaker_list) > 1

    with gr.Blocks() as demo:
        gr.Markdown("<h1 align='center'>EveryVoice StyleTTS2 Demo</h1>")
        with gr.Row():
            with gr.Column():
                inp_text = gr.Text(
                    placeholder="This text will be turned into speech.",
                    label="Input Text",
                )
                inputs = [inp_text]

                if has_speakers:
                    inp_speaker = gr.Dropdown(
                        choices=speaker_list,
                        value=speaker_list[0][1],
                        interactive=interactive_speaker,
                        label="Speaker",
                    )
                    inputs.append(inp_speaker)
                else:
                    synthesize_fn = partial(synthesize_fn, speaker=None)

                inp_reference = gr.Audio(
                    value=(
                        str(default_reference)
                        if (not has_speakers and default_reference)
                        else None
                    ),
                    label=(
                        "Override Reference Audio (optional)"
                        if has_speakers
                        else "Reference Audio"
                    ),
                    type="filepath",
                )
                inputs.append(inp_reference)

                with gr.Accordion("Advanced synthesis options", open=False):
                    inp_diffusion_steps = gr.Slider(
                        1, 20, value=5, step=1, label="Diffusion Steps"
                    )
                    inp_embedding_scale = gr.Slider(
                        0.1, 3.0, value=1.0, step=0.1, label="Embedding Scale"
                    )
                    inp_acoustic_blend = gr.Slider(
                        0.0, 1.0, value=0.3, step=0.05, label="Acoustic Blend"
                    )
                    inp_prosody_blend = gr.Slider(
                        0.0, 1.0, value=0.7, step=0.05, label="Prosody Blend"
                    )
                inputs.extend(
                    [
                        inp_diffusion_steps,
                        inp_embedding_scale,
                        inp_acoustic_blend,
                        inp_prosody_blend,
                    ]
                )
                btn = gr.Button("Synthesize")
            with gr.Column():
                out_audio = gr.Audio(format="wav", label="Output Audio")

        btn.click(synthesize_fn, inputs=inputs, outputs=[out_audio])
    return demo


def create_demo_app_styletts2(
    model_path: Path,
    output_dir: Path,
    speakers: "dict[str, Path]",
    default_reference: "Path | None" = None,
    accelerator: str = "auto",
    allowlist: list[str] = [],
    denylist: list[str] = [],
) -> "gr.Blocks":
    """Load a StyleTTS2 model and return a Gradio Blocks demo app.

    ``speakers`` maps display names to reference audio paths; their style
    encodings are pre-computed at startup so each synthesis call is fast.
    When ``speakers`` is empty the demo falls back to reference-upload mode
    using ``default_reference`` as the pre-populated audio widget value.
    """
    from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.cli.synthesize import (
        load_reference_style,
        load_styletts2_model,
    )
    from everyvoice.utils.heavy import get_device_from_accelerator

    require_ffmpeg()
    device = get_device_from_accelerator(accelerator)

    model, mel_transform = load_styletts2_model(model_path, device)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Pre-compute style encodings for each named speaker
    speaker_ref_s: dict[str, torch.Tensor] = {}
    for display_name, audio_path in speakers.items():
        logger.info(
            f"Pre-computing style encoding for speaker '{display_name}' from {audio_path}"
        )
        speaker_ref_s[display_name] = load_reference_style(
            model, mel_transform, audio_path, device
        )

    # Pre-compute the default reference encoding so synthesis never re-reads disk
    default_ref_s: "torch.Tensor | None" = None
    if default_reference is not None and default_reference.exists():
        logger.info(
            f"Pre-computing style encoding for default reference {default_reference}"
        )
        default_ref_s = load_reference_style(
            model, mel_transform, default_reference, device
        )

    norm_allowlist = [normalize_text(w) for w in allowlist]
    norm_denylist = [normalize_text(w) for w in denylist]

    synthesize_fn = partial(
        synthesize_audio_styletts2,
        module=model,
        mel_transform=mel_transform,
        device=device,
        output_dir=output_dir,
        speaker_ref_s=speaker_ref_s,
        default_ref_s=default_ref_s,
        allowlist=norm_allowlist,
        denylist=norm_denylist,
    )

    speaker_list: GradioChoices = [(name, name) for name in speakers]
    return make_gradio_display_styletts2(
        synthesize_fn,
        speaker_list,
        default_reference=default_reference if not speakers else None,
    )


def create_demo_app(
    text_to_spec_model_path: os.PathLike,
    spec_to_wav_model_path: os.PathLike,
    languages: list[str],
    speakers: list[str],
    outputs: list,  # list[str | AllowedDemoOutputFormats]
    output_dir: Path,
    accelerator: str,
    allowlist: list[str] = [],
    denylist: list[str] = [],
    app_ui_config: dict | None = None,
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
        app_ui_config=app_ui_config,
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
