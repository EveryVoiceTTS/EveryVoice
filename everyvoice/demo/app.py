import os
from functools import partial

import gradio as gr

from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.synthesize import (
    synthesize,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.prediction_writing_callback import (
    PredictionWritingWavCallback,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions import (
    SynthesizeOutputFormats,
)

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"


def synthesize_audio(
    text,
    duration_control,
    text_to_spec_model=None,
    spec_to_wav_model=None,
    language=None,
    speaker=None,
    output_dir=None,
):
    config, device, predictions = synthesize(
        model_path=text_to_spec_model,
        texts=[text],
        language=language,
        accelerator="cpu",
        devices="1",
        output_type=SynthesizeOutputFormats.wav,
        text_representation=TargetTrainingTextRepresentationLevel.characters,
        output_dir=output_dir,
        speaker=speaker,
        duration_control=1.0,
        filelist=None,
        teacher_forcing_directory=None,
        batch_size=1,
        num_workers=1,
        vocoder_path=spec_to_wav_model,
    )
    wav_writer = PredictionWritingWavCallback(
        output_dir=output_dir,
        config=config,
        output_key="postnet_output",
        device=device,
        global_step=1,
    )
    wav, sr = wav_writer.synthesize_audio(predictions[0])
    return sr, wav[0]


def create_demo_app(
    text_to_spec_model, spec_to_wav_model, language, speaker, output_dir
) -> gr.Interface:
    return gr.Interface(
        partial(
            synthesize_audio,
            text_to_spec_model=text_to_spec_model,
            spec_to_wav_model=spec_to_wav_model,
            language=language,
            speaker=speaker,
            output_dir=output_dir,
        ),
        [
            "textbox",
            gr.Slider(0.75, 1.5, 1.0, step=0.25),
        ],
        gr.Audio(format="mp3"),
        title="EveryVoice Demo",
    )
