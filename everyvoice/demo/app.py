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
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions import (
    SynthesizeOutputFormats,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
    load_hifigan_from_checkpoint,
)
from everyvoice.utils.heavy import get_device_from_accelerator

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"


def synthesize_audio(
    text,
    duration_control,
    text_to_spec_model,
    vocoder_model,
    vocoder_config,
    accelerator,
    device,
    language=None,
    speaker=None,
    output_dir=None,
):
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
    output_key = "postnet_output" if text_to_spec_model.config.model.use_postnet else "output"
    wav_writer = PredictionWritingWavCallback(
        output_dir=output_dir,
        config=config,
        output_key=output_key,
        device=device,
        global_step=1,
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
    language,
    speaker,
    output_dir,
    accelerator,
) -> gr.Interface:
    device = get_device_from_accelerator(accelerator)
    vocoder_ckpt = torch.load(spec_to_wav_model_path, map_location=device)
    vocoder_model, vocoder_config = load_hifigan_from_checkpoint(vocoder_ckpt, device)
    model: FastSpeech2 = FastSpeech2.load_from_checkpoint(text_to_spec_model_path).to(
        device
    )
    model.eval()
    return gr.Interface(
        partial(
            synthesize_audio,
            text_to_spec_model=model,
            vocoder_model=vocoder_model,
            vocoder_config=vocoder_config,
            language=language,
            speaker=speaker,
            output_dir=output_dir,
            accelerator=accelerator,
            device=device,
        ),
        [
            "textbox",
            gr.Slider(0.75, 1.75, 1.0, step=0.25),
        ],
        gr.Audio(format="mp3"),
        title="EveryVoice Demo",
    )
