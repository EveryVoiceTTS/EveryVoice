# Customize to your language

## Step 1: Make sure you have Permission!

So, you want to build a text-to-speech system for a new language or dataset - cool! But, just because you **can** build a text-to-speech system, doesn't mean you **should**. There are a lot of important ethical questions around text-to-speech. For example, it's not ethical to just use audio you find somewhere online if it doesn't have explicit permission to use it for the purposes of text-to-speech. The first step is always to make sure you have permission to use the data in question and that whoever contributed their voice to the data you want to use is aware and supportive of your goal.

Creating a text-to-speech model without permission is unethical, but even when you do have permission, you should take great care in how you distribute the model you have created. Increasingly, text-to-speech technology is used in fraud, unauthorized impersonation. The technology has also been used to disenfranchise voice actors and other professionals. When you create an EveryVoice model, you are responsible for ensuring the model is only used and distributed according to the permissions you have. To help with this accountability, you will be required by EveryVoice to provide a full name and contact information that will also be distributed with the model.

## Step 2: Gather Your Data

The first thing to do is to get all the data you have (in this case audio with text transcripts) together in one place. Your audio should be in 'wav' format. Ideally it would be 16bit, mono (one channel) audio sampled somewhere between 22.05kHz and 48kHz. If that doesn't mean anything to you, don't worry, we can ensure the right format in later steps.
It's best if your audio clips are somewhere between half a second and 10 seconds long. Any longer and it could be difficult to train. If your audio is longer than this, we suggest processing it into smaller chunks first.

Your text should be consistently written and should be in a pipe-separated values spreadsheet, similar to [this file](https://github.com/roedoejet/EveryVoice/blob/main/everyvoice/filelists/lj_full.psv). It should have a column that contains text and a column that contains the `basename` of your associated audio file. So if you have a recording of somebody saying "hello how are you?" and the corresponding audio is called `mydata0001.wav`
then you should have a psv file that looks like this:

```csv hl_lines="2"

basename|text
mydata0001|hello how are you?
mydata0002|some other sentence.
...
```

We also support comma and tab separated files, but recommend using pipes (|).

You can also use the "festival" format which is like this (example from [Sinhala TTS](https://openslr.org/30/)):

```text
( sin_2241_0329430812 " කෝකටත් මං වෙනදා තරම් කාලෙ ගන්නැතිව ඇඳ ගත්තා " )
( sin_2241_0598895166 " ඇන්ජලීනා ජොලී කියන්නේ පසුගිය දිනවල බොහෝ සෙයින් කතා බහට ලක්වූ චරිතයක් " )
( sin_2241_0701577369 " ආර්ථික චින්තනය හා සාමාජීය දියුණුව ඇති කළ හැකිවනුයේ පුද්ගල ආර්ථික දියුණුව සලසා දීමෙන්ය " )
( sin_2241_0715400935 " ඉන් අදහස් වන්නේ විචාරාත්මක විනිවිද දැකීමෙන් තොර බැල්මයි " )
( sin_2241_0817100025 " අප යුද්ධයේ පළමු පියවරේදීම පරාද වී අවසානය " )
```

In this format, there are corresponding wav files labelled sin_2241_0329430812.wav etc..

## Step 3: Install EveryVoice

Head over to the [install documentation](../install.md) and install EveryVoice

## Step 4: Run the Configuration Wizard 🧙

Once you have your data, the best thing to do is to run the Configuration Wizard 🧙 which will help you configure a new project. To do that run:

```bash
everyvoice new-project
```

After running the wizard, cd into your newly created directory. Let's call it `test` for now.

```bash
cd test
```

## Step 5: Run the Preprocessor

Your models need to do a number of preprocessing steps in order to prepare for training. To preprocess everything you need, run the following:

```bash
everyvoice preprocess config/{{ config_filename('text-to-spec') }}
```

## Step 6: Select a Vocoder

So you don't need to train your own vocoder, EveryVoice has a variety of publicly released vocoders available [here](TODO). Follow the instructions there for downloading the checkpoints.

EveryVoice is also compatible out-of-the-box with the UNIVERSAL_V1 HiFiGAN checkpoint, which is very good quality. You can find it [here](https://github.com/jik876/hifi-gan?tab=readme-ov-file#pretrained-model).

Using a pre-trained vocoder is recommended, and the above checkpoints should work well even for new languages.

### Train your own Vocoder

You might want to train your own vocoder, but this takes a long time (up to 2 weeks on a single GPU), uses a lot of electricity, and unless you know what you are doing, you are unlikely to improve upon the publicly available models discussed above, even for a new language.

```bash
everyvoice train spec-to-wav config/{{ config_filename('spec-to-wav') }}
```

By default, we run our training with PyTorch Lightning's "auto" strategy. But, if you are on a machine where you know the hardware, you can specify it like:

```bash
everyvoice train spec-to-wav config/{{ config_filename('spec-to-wav') }} -d 1 -a gpu
```

Which would use the GPU accelerator and specify 1 device/chip.

## Step 7: Train your Feature Prediction Network

To generate audio when you train your feature prediction network, you need to add your vocoder checkpoint to the `config/{{ config_filename('text-to-spec') }}`

At the bottom of that file you'll find a key called `vocoder_path`. Add the absolute path to your trained vocder (here it would be `/path/to/test/logs_and_checkpoints/VocoderExperiment/base/checkpoints/last.ckpt` where `/path/to` would be the actual path to it on your computer.)

Once you've replaced the vocoder_path key, you can train your feature prediction network:

```bash
everyvoice train text-to-spec config/{{ config_filename('text-to-spec') }}
```

## Step 8: Synthesize Speech in Your Language!

You can synthesize by pointing the CLI to your trained feature prediction network and passing in the text. You can export the wav or spectrogram (pt) files.

```bash
everyvoice synthesize from-text logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt -t "මෙදා සැරේ සාකච්ඡාවක් විදියට නෙවෙයි නේද පල කරල තියෙන්නෙ" -a gpu -d 1 --output-type wav
```

<!-- % Step 10 (optional): Finetune your vocoder

% ----------------------------------------

% .. code-block:: bash

% everyvoice train text-to-wav config/{{ config_filename('text-to-wav') }}

% Step 11: Synthesize Speech

% --------------------------

% .. code-block:: bash

% everyvoice synthesize from-text -t "hello world" -c config/{{ config_filename('text-to-wav') }}

% .. warning::

% TODO: this doesn't exist yet

% TODO: e2e needs checkpoint paths -->
