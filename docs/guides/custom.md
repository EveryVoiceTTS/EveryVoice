# Customize to your language

## Step 1: Make sure you have Permission!

So, you want to build a text-to-speech system for a new language or dataset - cool! But, just because you _can_ build a text-to-speech system, doesn't mean you _should_. There are a lot of important ethical questions around text-to-speech. For example, it's **not** ethical to just use audio you find somewhere online if it doesn't have explicit permission to use it for the purposes of text-to-speech. The first step is always to make sure you have permission to use the data in question and that whoever contributed their voice to the data you want to use is aware and supportive of your goal.

Creating a text-to-speech model without permission is unethical, but even when you do have permission, you should take great care in how you distribute the model you have created. Increasingly, text-to-speech technology is used in fraud and unauthorized impersonation. The technology has also been used to disenfranchise voice actors and other professionals. When you create an EveryVoice model, you are responsible for ensuring the model is only used and distributed according to the permissions you have. To help with this accountability, you will be required by EveryVoice to attest that you have permission to use your data and to provide a full name and contact information that will also be distributed with the model.

In addition, we invite you to check out our [short guide](./ethics.md) that contains prompts about ethical questions _before_ starting on any of the next steps.

## Step 2: Gather Your Data

The first thing to do is to get all the data you have (in this case audio with text transcripts) together in one place. Your audio should be in a lossless 'wav' format. Ideally it would be 16bit, mono (one channel) audio sampled somewhere between 22.05kHz and 48kHz. If that doesn't mean anything to you, don't worry, we can ensure the right format in later steps.

It's best if your audio clips are somewhere between half a second and 10 seconds long. Any longer and it could be difficult to train depending on the size of your GPU. If your audio is significantly longer than this, we suggest processing it into smaller chunks first. To do this, you can use the `everyvoice segment` command. For this to work you need a plain text transcript and some corresponding audio. You can then run the segmenter: `everyvoice segment align path_to_text.txt path_to_audio.wav`. You can then install [Praat](https://www.fon.hum.uva.nl/praat/) and use it to inspect the .TextGrid file that was generated, and adjust any alignments as necessary. Once you are happy with your alignments, you can use `everyvoice segment extract path_to_alignment.TextGrid path_to_audio.wav outdir` which will then create a folder called `outdir` with your audio, and a metadata file containing references to each of your audio files and the corresponding text.

Your text should be consistently written and should be in a pipe-separated values spreadsheet, similar to [this file](https://github.com/EveryVoiceTTS/EveryVoice/blob/main/everyvoice/filelists/lj_full.psv). It should have a column that contains text and a column that contains the `basename` of your associated audio file. So if you have a recording of somebody saying "hello how are you?" and the corresponding audio is called `mydata0001.wav`
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

Head over to the [installation documentation](../install.md) and install EveryVoice

## Step 4: Run the Configuration Wizard 🧙

Once you have your data, the best thing to do is to run the Configuration Wizard 🧙 which will help you configure a new project. To do that run:

```bash
everyvoice new-project
```

After running the wizard, cd into your newly created directory. Let's call it `<your_everyvoice_project>` for now.

```bash
cd your_everyvoice_project
```

!!! important
    After you run the Configuration Wizard 🧙, please inspect your text configuration `config/{{ config_filename('text') }}` to make sure everything looks right. That is, if some unexpected symbols show up, please inspect your data (if you remove symbols from the configuration here, they will be ignored during training). Sometimes characters that are treated as punctuation by default will need to be removed from the punctuation list if they are treated as non-punctuation in your language.

## Step 5: Choose a Model

EveryVoice supports two model architectures, and the right choice depends on what you want to prioritize. This section explains the difference so you know what you're getting into.

**[FastSpeech2](./fastspeech2.md)** is a two-stage pipeline: a feature prediction network converts text to mel spectrograms, and then a separate vocoder converts those spectrograms to audio. It trains relatively quickly (usually less than a day for most datasets on a single GPU), and produces good, consistent speech. If this is your first time training a TTS model with EveryVoice, or if you want reliable results without a lot of fuss, this is a good place to start. Even if you plan on training a StyleTTS2 model, this is a good low-cost option for training a model with your data. If your system isn't intelligible with FastSpeech2, it certainly won't work with StyleTTS2, and the solution is likely to be found in fixing issues in your data. If your FastSpeech2 system sounds pretty good, and you want to take it to the next level, try StyleTTS2.

**[StyleTTS2](./styletts2.md)** is an end-to-end model that goes directly from text to audio in a single pass, using a diffusion process to model speaking style and other unlabelled variation (i.e. recording environment, minor dialect differences). At synthesis time it takes a short reference audio clip and uses it to match the style. Training is split into two stages and it takes a lot longer than FastSpeech2 (over a week with multiple GPUs on most datasets), but can produce particularly natural-sounding speech, especially for expressive or varied speaking styles.

If you're not sure which to pick, start with FastSpeech2.

- Continue with **[FastSpeech2 →](./fastspeech2.md)**
- Continue with **[StyleTTS2 →](./styletts2.md)**
