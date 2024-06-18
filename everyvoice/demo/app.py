import os
from functools import partial

import gradio as gr
import torch

from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.synthesize import (
    synthesize_helper,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
    FastSpeech2,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.prediction_writing_callback import (
    PredictionWritingWavCallback,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
    load_hifigan_from_checkpoint,
)
from everyvoice.utils.heavy import get_device_from_accelerator

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"


def synthesize_audio(
    text,
    duration_control,
    language,
    speaker,
    text_to_spec_model,
    vocoder_model,
    vocoder_config,
    accelerator,
    device,
    output_dir=None,
):
    if text == "":
        raise gr.Error(
            "Text for synthesis was not provided. Please type the text you want to be synthesized into the textfield."
        )
    if language is None:
        raise gr.Error("Language is not selected. Please select a language.")
    if speaker is None:
        raise gr.Error("Speaker is not selected. Please select a speaker.")
    config, device, predictions = synthesize_helper(
        model=text_to_spec_model,
        vocoder_model=vocoder_model,
        vocoder_config=vocoder_config,
        texts=[text],
        language=language,
        accelerator=accelerator,
        devices="1",
        device=device,
        global_step=1,
        vocoder_global_step=1,  # dummy value since the vocoder step is not used
        output_type=[],
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
        global_step=1,
        vocoder_global_step=1,  # dummy value since the vocoder step is not used
        vocoder_model=vocoder_model,
        vocoder_config=vocoder_config,
    )
    # move to device because lightning accumulates predictions on cpu
    predictions[0][output_key] = predictions[0][output_key].to(device)
    wav, sr = wav_writer.synthesize_audio(predictions[0])
    return sr, wav[0]


def create_demo_app(
    text_to_spec_model_path,
    spec_to_wav_model_path,
    output_dir,
    accelerator,
) -> gr.Blocks:
    device = get_device_from_accelerator(accelerator)
    vocoder_ckpt = torch.load(spec_to_wav_model_path, map_location=device)
    vocoder_model, vocoder_config = load_hifigan_from_checkpoint(vocoder_ckpt, device)
    model: FastSpeech2 = FastSpeech2.load_from_checkpoint(text_to_spec_model_path).to(
        device
    )
    model.eval()
    synthesize_audio_preset = partial(
        synthesize_audio,
        text_to_spec_model=model,
        vocoder_model=vocoder_model,
        vocoder_config=vocoder_config,
        output_dir=output_dir,
        accelerator=accelerator,
        device=device,
    )
    lang_list = list(model.lang2id.keys())
    speak_list = list(model.speaker2id.keys())
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
                    inp_lang = gr.Dropdown(lang_list, label="Language")
                    inp_speak = gr.Dropdown(speak_list, label="Speaker")
                btn = gr.Button("Synthesize")
            with gr.Column():
                out_audio = gr.Audio(format="mp3")
        btn.click(
            synthesize_audio_preset,
            inputs=[inp_text, inp_slider, inp_lang, inp_speak],
            outputs=[out_audio],
        )
    return demo
