import os
import string
import subprocess
import sys
from functools import partial
from unicodedata import normalize

import gradio as gr
import torch
from loguru import logger

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
    load_hifigan_from_checkpoint,
)
from everyvoice.utils import slugify
from everyvoice.utils.heavy import get_device_from_accelerator

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"


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
        teacher_forcing_directory=None,
        batch_size=1,
        num_workers=1,
    )

    wav_writer = callbacks[SynthesizeOutputFormats.wav]
    wav_output = wav_writer.get_filename(basename, speaker, language)

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
) -> gr.Blocks:
    # Early argument validation where possible
    possible_outputs = [x.value for x in SynthesizeOutputFormats]

    # this used to be `if outputs == ["all"]:` but my Enum() constructor for
    # AllowedDemoOutputFormats breaks that, unfortunately, and enum.StrEnum
    # doesn't appear until Python 3.11 so I can't use it.
    if len(outputs) == 1 and getattr(outputs[0], "value", outputs[0]) == "all":
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

    require_ffmpeg()
    device = get_device_from_accelerator(accelerator)
    vocoder_ckpt = torch.load(
        spec_to_wav_model_path, map_location=device, weights_only=True
    )
    # TODO: Should we also wrap this load_hifigan_from_checkpoint in case the checkpoint is not a Vocoder?
    vocoder_model, vocoder_config = load_hifigan_from_checkpoint(vocoder_ckpt, device)
    model: FastSpeech2 = FastSpeech2.load_from_checkpoint(text_to_spec_model_path).to(  # type: ignore
        device
    )
    model.eval()
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
    lang_list = []
    speak_list = []
    if languages == ["all"]:
        lang_list = model_languages
    else:
        for language in languages:
            if language in model_languages:
                lang_list.append(language)
            else:
                print(
                    f"Attention: The model have not been trained for speech synthesis in '{language}' language. The '{language}' language option will not be available for selection."
                )
    if speakers == ["all"]:
        speak_list = model_speakers
    else:
        for speaker in speakers:
            if speaker in model_speakers:
                speak_list.append(speaker)
            else:
                print(
                    f"Attention: The model have not been trained for speech synthesis with '{speaker}' speaker. The '{speaker}' speaker option will not be available for selection."
                )

    if lang_list == []:
        raise ValueError(
            f"Language option has been activated, but valid languages have not been provided. The model has been trained in {model_languages} languages. Please select either 'all' or at least some of them."
        )
    if speak_list == []:
        raise ValueError(
            f"Speaker option has been activated, but valid speakers have not been provided. The model has been trained with {model_speakers} speakers. Please select either 'all' or at least some of them."
        )
    default_lang = lang_list[0]
    interactive_lang = len(lang_list) > 1
    default_speak = speak_list[0]
    interactive_speak = len(speak_list) > 1
    default_output = output_list[0]
    interactive_output = len(output_list) > 1
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            <h1 align="center">EveryVoice Demo</h1>
            """
        )
        with gr.Row():
            with gr.Column():
                inp_text = gr.Text(
                    placeholder="This text will be turned into speech.",
                    label="Input Text",
                )
                inp_slider = gr.Slider(
                    0.75, 1.75, 1.0, step=0.25, label="Duration Multiplier"
                )
                with gr.Row():
                    inp_lang = gr.Dropdown(
                        choices=lang_list,
                        value=default_lang,
                        interactive=interactive_lang,
                        label="Language",
                    )
                    inp_speak = gr.Dropdown(
                        choices=speak_list,
                        value=default_speak,
                        interactive=interactive_speak,
                        label="Speaker",
                    )
                inputs = [inp_text, inp_slider, inp_lang, inp_speak]
                if output_list != [SynthesizeOutputFormats.wav]:
                    with gr.Row():
                        output_format = gr.Dropdown(
                            choices=output_list,
                            value=default_output,
                            interactive=interactive_output,
                            label="Output Format",
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
                btn = gr.Button("Synthesize")
            with gr.Column():
                out_audio = gr.Audio(format="wav")
                if output_list == [SynthesizeOutputFormats.wav]:
                    # When the only output option is wav, don't show the File Output box
                    outputs = [out_audio]
                else:
                    out_file = gr.File(label="File Output", elem_id="file_output")
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
