# Training a StyleTTS2 Model

This page picks up from [Step 5: Choose a Model](./custom.md#step-5-choose-a-model) in the main guide. If you haven't completed [Steps 1–4 in the main guide](./custom.md) yet, start there first.

## Step 1: Fetch Pretrained Models

StyleTTS2 uses several pretrained components — an F0 extractor, an ASR aligner, a PLBERT language model, and a WavLM model — that are downloaded from HuggingFace before training begins. If you are on a cluster where your GPU nodes don't have internet access, you'll want to run this step on a login node first so the files are cached and available when training starts.

```bash
everyvoice fetch-pretrained text-to-wav config/{{ config_filename('text-to-wav') }}
```

If your GPU nodes do have internet access, you can skip this step and the files will be downloaded automatically when training starts.

## Step 2: Run the Preprocessor

Your model needs to do a number of preprocessing steps in order to prepare for training. To preprocess everything you need, run the following:

```bash
everyvoice preprocess text-to-wav config/{{ config_filename('text-to-wav') }}
```

!!! important
    StyleTTS2 training requires a GPU with CUDA. Make sure you are running the following steps on a machine with a compatible GPU.

### Step 2a: Inspect your symbol set

If loading your configuration failed with a `pydantic.ValidationError` mentioning you have some symbols that "are not present in the pretrained StyleTTS2 text-encoder symbol table", it's due to symbols in your data that are incompatible with StyleTTS2 — but it can be fixed! This can happen on any command that loads your `text-to-wav` config, including `everyvoice preprocess` above. If you didn't get this error, please proceed to Step 3.

Unlike EveryVoice's other models, StyleTTS2 uses a pretrained text encoder. That encoder has a fixed embedding table for a specific set of 178 symbols (letters, IPA phones, and punctuation) that it saw during training, and it cannot produce a meaningful embedding for a symbol it has never seen. Because of this, every symbol declared in your `text` configuration's `symbols` must also be a member of that pretrained set.

If your configuration declares a symbol outside that set, EveryVoice will refuse to load the configuration rather than silently proceeding to train a broken model. The error message tells you exactly which symbols are the problem, and, for each one, the closest pretrained symbol it found (using panphon's articulatory-feature distance for IPA phones, edit distance for multigraphs and a last-resort Unicode distance measure otherwise) along with a distance score, e.g.:

```
Value error, The following symbols declared in your TextConfig are not present in
the pretrained StyleTTS2 text-encoder symbol table: {'ʒ'}. Either remove them from
your TextConfig, use a custom pretrained_symbols list, or add 'to_replace' rules to
substitute them for symbols the pretrained encoder does know. Closest pretrained
symbols: 'ʒ' is closest to 'ʃ' (distance=0.25).
```

To fix your model, add a `to_replace` rule to your text configuration for each suggested symbol, e.g.:

```yaml
text:
  to_replace:
    "ʒ": "ʃ"
```

!!! note
    A suggested substitution is a starting point, always sanity-check that the suggested symbol is phonetically reasonable for your language before adopting it, while also making sure that you are not mapping multiple symbols to the same symbol. Some symbols may have no close match at all, so the mappings might look fairly bad. My experience is that the model is still able to recover from a few of these bad mappings.

## Step 3: Train your Model

StyleTTS2 training is split into two stages that must be run in order.

**Stage 1**

```bash
everyvoice train text-to-wav config/{{ config_filename('text-to-wav') }} --mode first
```

**Stage 2** Once Stage 1 has finished, run:

```bash
everyvoice train text-to-wav config/{{ config_filename('text-to-wav') }} --mode second
```

!!! tip
    While your model is training, you can use TensorBoard to view the logs, which will show information about the progress of training. To use TensorBoard, make sure that your conda environment is activated and run `tensorboard --logdir path/to/logs_and_checkpoints`. Then your logs will be viewable at [http://localhost:6006](http://localhost:6006).

By default, training uses PyTorch Lightning's "auto" strategy. If you are on a machine where you know the hardware, you can specify it:

```bash
everyvoice train text-to-wav config/{{ config_filename('text-to-wav') }} --mode first -d 4 -a gpu
```

Which would use the GPU accelerator (`-a gpu`) and specify 4 devices/chips (`-d 4`).

## Step 4: Synthesize Speech in Your Language!

StyleTTS2 generates speech by sampling a speaking style from a short reference audio clip. The reference audio should be a clean recording of a few seconds, but it doesn't need to match the text you want to synthesize. You can use any recording from your training data as a reference.

```bash
everyvoice synthesize text-to-wav logs_and_checkpoints/E2E-Experiment/checkpoints/last.ckpt \
    --reference path/to/reference.wav \
    --text "your text here"
```

You can pass `--text` multiple times to synthesize several utterances at once:

```bash
everyvoice synthesize text-to-wav logs_and_checkpoints/E2E-Experiment/checkpoints/last.ckpt \
    --reference path/to/reference.wav \
    --text "First sentence." \
    --text "Second sentence."
```

## Optional: Evaluation

If you want to evaluate the model you just built, you can make use of the `everyvoice evaluate` command. In order to use it, you have to first generate some audio (see Step 5) and then you can evaluate either a single file with `everyvoice evaluate -f your_file.wav` or a directory of audio files with `everyvoice evaluate -d path_to_wavs/`. This will report predictions for three metrics: Wideband Perceptual Estimation of Speech Quality (PESQ), Short-Time Objective Intelligibility (STOI), and Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) using the model described in [this](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10096680) paper. You can also provide a non-matching reference to predict a Mean Opinion Score (MOS) for your generated audio: `everyvoice evaluate  -d path_to_wavs/ -r path_to_reference.wav`. The reference should be a path to non-generated, good quality audio but it doesn't need to match the exact utterance that was generated.

Please refer to `everyvoice evaluate --help` for more information.

!!! note
    Automatic evaluation can be helpful, but please take the reported numbers with a grain of salt. They are not always reliable, and do not always correlate well with human judgements.
