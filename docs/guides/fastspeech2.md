# Training a FastSpeech2 Model

This page picks up from [Step 5: Choose a Model](./custom.md#step-5-choose-a-model) in the main guide. If you haven't completed [Steps 1–4 in the main guide](./custom.md) yet, start there first.

## Step 1: Run the Preprocessor

Your model needs to do a number of preprocessing steps in order to prepare for training. To preprocess everything you need, run the following:

```bash
everyvoice preprocess text-to-spec config/{{ config_filename('text-to-spec') }}
```

## Step 2: Select a Vocoder

You do not need to train your own vocoder.
EveryVoice is compatible with the UNIVERSAL_V1 HiFiGAN checkpoint from [the official HiFiGAN implementation](https://github.com/jik876/hifi-gan?tab=readme-ov-file#pretrained-model), which is very good quality. You can find the EveryVoice-compatible version of this checkpoint [here](https://drive.google.com/drive/folders/1ya0U4K2d26DoJamg96cEynMJ1w1Tm8nU?usp=sharing).

You can download the checkpoint by following the link above, or you can download it using the command line with [gdown](https://pypi.org/project/gdown/). First ensure that you have _gdown_ installed; you can install it with `pip install gdown`. Then to download the checkpoint you can run:

```bash
gdown https://drive.google.com/uc?id=1-iarZV2hTeociQjTX7l2WShuXalROGQf
```

Using a pre-trained vocoder is recommended, and the above checkpoint should work well even for new languages after [finetuning](./finetune.md).

### Train your own Vocoder

You might want to train your own vocoder, but this takes a long time (up to 2 weeks on a single GPU), uses a lot of electricity, and unless you know what you are doing, you are unlikely to improve upon the publicly available models discussed above, even for a new language. So we do not recommend it. You are almost always better off just using the pre-trained vocoder and then [finetuning](./finetune.md) on the predictions from your feature prediction network. If you really do want to train your own vocoder though, you can run the following command:

```bash
everyvoice train spec-to-wav config/{{ config_filename('spec-to-wav') }}
```

By default, we run our training with PyTorch Lightning's "auto" strategy. But, if you are on a machine where you know the hardware, you can specify it like:

```bash
everyvoice train spec-to-wav config/{{ config_filename('spec-to-wav') }} -d 1 -a gpu
```

Which would use the GPU accelerator (`-a gpu`) and specify 1 device/chip (`-d 1`).

## Step 3: Train your Feature Prediction Network

To generate audio when you train your feature prediction network, you need to add your vocoder checkpoint to the `config/{{ config_filename('text-to-spec') }}`

Then, you can train your feature prediction network:

```bash
everyvoice train text-to-spec config/{{ config_filename('text-to-spec') }}
```

!!! tip
    While your model is training, you can use TensorBoard to view the logs which will show information about the progress of training and display spectrogram images. At the bottom of the `config/{{ config_filename('text-to-spec') }}` file you'll find a key called `vocoder_path`. Add the absolute path to your trained vocoder (here it would be `/path/to/test/logs_and_checkpoints/VocoderExperiment/base/checkpoints/last.ckpt` where `/path/to` would be the actual path to it on your computer.) If you provide `vocoder_path` key, then you will also be able to hear audio in the logs. To use TensorBoard, make sure that your conda environment is activated and run `tensorboard --logdir path/to/logs_and_checkpoints`. Then your logs will be viewable at [http://localhost:6006](http://localhost:6006).

## Step 4 (optional): Finetune your Vocoder

When you have finished training your Feature Prediction Network, we recommend [finetuning](./finetune.md) your vocoder. This step is optional, but it will help get rid of metallic artefacts that are often present if you don't finetune your vocoder. Note, it will likely not help with any mispronounciations. If you notice these types of errors, it is likely due to issues with the training data (e.g. too much variation in pronunciation or recording quality in the dataset, or discrepancies between the recording and transcription.)

## Step 5: Synthesize Speech in Your Language!

#### Command Line

You can synthesize by pointing the CLI to your trained feature prediction network and passing in the text. You can export the wav or spectrogram (pt) files.

```bash
everyvoice synthesize from-text logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt -t "මෙදා සැරේ සාකච්ඡාවක් විදියට නෙවෙයි නේද පල කරල තියෙන්නෙ" -a gpu -d 1 --output-type wav
```

#### Demo App

You can also synthesize audio by starting up the EveryVoice Demo using your Feature Prediction and Vocoder checkpoints:

```bash
everyvoice demo logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt logs_and_checkpoints/VocoderExperiment/base/checkpoints/last.ckpt
```

And an interactive demo will be available at [http://localhost:7260](http://localhost:7260)

Please consult the demo help for more information on how to use the demo: `everyvoice demo --help`.

You can provide a custom json configuration file to override parts of the user interface the demo by using the `--ui-config-file` or `-C` flag, e.g. `everyvoice demo --ui-config-file=my_config.json  logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt logs_and_checkpoints/VocoderExperiment/base/checkpoints/last.ckpt`.

This user interface configuration can provide a custom title (`app_title`), and custom labels for other UI elements as well as display for the languages and speakers. See sample below:

```json
{
  "app_title": "My Custom TTS Demo",
  "app_description": "This is a custom text-to-speech demo for my language.",
  "app_instructions": "Type your text in the box below and click 'Generate speech' to generate it spoken in the selected language and by the selected speaker.",
  "languages": {
    "en": "English",
    "fr": "French"
  },
  "speakers": {
    "speaker1": "Speaker One",
    "speaker2": "Speaker Two"
  },
  "input_text_label": "Input Text",
  "duration_multiplier_label": "Duration Multiplier",
  "language_label": "Language",
  "speaker_label": "Speaker",
  "output_format_label": "Output Format",
  "synthesize_label": "Synthesize",
  "file_output_label": "File Output"
}
```

## Optional: Evaluation

If you want to evaluate the model you just built, you can make use of the `everyvoice evaluate` command. In order to use it, you have to first generate some audio (see Step 5) and then you can evaluate either a single file with `everyvoice evaluate -f your_file.wav` or a directory of audio files with `everyvoice evaluate -d path_to_wavs/`. This will report predictions for three metrics: Wideband Perceptual Estimation of Speech Quality (PESQ), Short-Time Objective Intelligibility (STOI), and Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) using the model described in [this](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10096680) paper. You can also provide a non-matching reference to predict a Mean Opinion Score (MOS) for your generated audio: `everyvoice evaluate  -d path_to_wavs/ -r path_to_reference.wav`. The reference should be a path to non-generated, good quality audio but it doesn't need to match the exact utterance that was generated.

Please refer to `everyvoice evaluate --help` for more information.

!!! note
    Automatic evaluation can be helpful, but please take the reported numbers with a grain of salt. They are not always reliable, and do not always correlate well with human judgements.
