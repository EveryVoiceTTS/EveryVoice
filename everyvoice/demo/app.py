import os
import string
import subprocess
import sys
from functools import partial
from unicodedata import normalize

import gradio as gr
import torch
import torchaudio
from loguru import logger

from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.synthesize import (
    synthesize_helper,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
    FastSpeech2,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.prediction_writing_callback import (
    PredictionWritingOfflineRASCallback,
    PredictionWritingReadAlongCallback,
    PredictionWritingSpecCallback,
    PredictionWritingTextGridCallback,
    PredictionWritingWavCallback,
    get_tokens_from_duration_and_labels,
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
    text_to_spec_model,
    vocoder_model,
    vocoder_config,
    accelerator,
    device,
    allowlist,
    denylist,
    output_dir=None,
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
        raise gr.Error("Speaker is not selected. Please select an output format.")
    config, device, predictions = synthesize_helper(
        model=text_to_spec_model,
        vocoder_model=vocoder_model,
        vocoder_config=vocoder_config,
        texts=[text],
        language=language,
        accelerator=accelerator,
        devices="1",
        device=device,
        global_step=text_to_spec_model.config.training.max_steps,
        vocoder_global_step=vocoder_model.config.training.max_steps,
        output_type=[output_format],
        text_representation=TargetTrainingTextRepresentationLevel.characters,
        output_dir=output_dir,
        speaker=speaker,
        duration_control=duration_control,
        filelist=None,
        teacher_forcing_directory=None,
        batch_size=1,
        num_workers=1,
    )
    output_key = (
        "postnet_output" if text_to_spec_model.config.model.use_postnet else "output"
    )
    wav_writer = PredictionWritingWavCallback(
        output_dir=output_dir,
        config=config,
        output_key=output_key,
        device=device,
        global_step=text_to_spec_model.config.training.max_steps,
        vocoder_global_step=vocoder_model.config.training.max_steps,
        vocoder_model=vocoder_model,
        vocoder_config=vocoder_config,
    )
    # move to device because lightning accumulates predictions on cpu
    predictions[0][output_key] = predictions[0][output_key].to(device)
    wav, sr = wav_writer.synthesize_audio(predictions[0])
    torchaudio.save(
        wav_writer.get_filename(basename, speaker, language),
        # the vocoder output includes padding so we have to remove that
        wav[0],
        sr,
        format="wav",
        encoding="PCM_S",
        bits_per_sample=16,
    )
    wav_output = wav_writer.get_filename(basename, speaker, language)
    file_writer = None
    file_output = None
    if output_format == SynthesizeOutputFormats.readalong_html.name:
        file_writer = PredictionWritingOfflineRASCallback(
            config=config,
            global_step=text_to_spec_model.config.training.max_steps,
            output_dir=output_dir,
            output_key=output_key,
            wav_callback=wav_writer,
        )

    if output_format == SynthesizeOutputFormats.readalong_xml.name:
        file_writer = PredictionWritingReadAlongCallback(
            config=config,
            global_step=text_to_spec_model.config.training.max_steps,
            output_dir=output_dir,
            output_key=output_key,
        )

    if output_format == SynthesizeOutputFormats.spec.name:
        file_writer = PredictionWritingSpecCallback(
            config=config,
            global_step=text_to_spec_model.config.training.max_steps,
            output_dir=output_dir,
            output_key=output_key,
        )

    if output_format == SynthesizeOutputFormats.textgrid.name:
        file_writer = PredictionWritingTextGridCallback(
            config=config,
            global_step=text_to_spec_model.config.training.max_steps,
            output_dir=output_dir,
            output_key=output_key,
        )
    if file_writer is not None:
        max_seconds, phones, words = get_tokens_from_duration_and_labels(
            predictions[0]["duration_prediction"][0],
            predictions[0]["text_input"][0],
            text,
            text_to_spec_model.text_processor,
            text_to_spec_model.config,
        )

        file_writer.save_aligned_text_to_file(
            basename=basename,
            speaker=speaker,
            language=language,
            max_seconds=max_seconds,
            phones=phones,
            words=words,
        )
        file_output = file_writer.get_filename(basename, speaker, language)

    return wav_output, file_output


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
    text_to_spec_model_path,
    spec_to_wav_model_path,
    languages,
    speakers,
    outputs,
    output_dir,
    accelerator,
    allowlist: list[str] = [],
    denylist: list[str] = [],
) -> gr.Blocks:
    require_ffmpeg()
    device = get_device_from_accelerator(accelerator)
    vocoder_ckpt = torch.load(spec_to_wav_model_path, map_location=device)
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
    possible_outputs = [x.name for x in SynthesizeOutputFormats]
    lang_list = []
    speak_list = []
    output_list = []
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
    if outputs == ["all"]:
        output_list = possible_outputs
    else:
        for output in outputs:
            if output in possible_outputs:
                output_list.append(output)
            else:
                print(
                    f"Attention: This model is not able to produce '{output}' as an output. The '{output}' option will not be available for selection. Please choose from the following possible outputs: {', '.join(possible_outputs)}"
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
                with gr.Row():
                    output_format = gr.Dropdown(
                        choices=output_list,
                        value=default_output,
                        interactive=interactive_output,
                        label="Output Format",
                    )
                btn = gr.Button("Synthesize")
            with gr.Column():
                out_audio = gr.Audio(format="wav")
                out_file = gr.File(label="File Output")
        btn.click(
            synthesize_audio_preset,
            inputs=[inp_text, inp_slider, inp_lang, inp_speak, output_format],
            outputs=[out_audio, out_file],
        )
    return demo
