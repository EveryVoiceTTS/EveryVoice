# EveryVoice TTS Toolkit 💬

[![codecov](https://codecov.io/gh/roedoejet/EveryVoice/branch/main/graph/badge.svg?token=yErCxf64IU)](https://codecov.io/gh/roedoejet/EveryVoice)
[![license](https://img.shields.io/badge/Licence-MIT-green)](LICENSE)

This is an implementation of the Text-to-Speech (TTS) model used by the Small Teams "Speech Generation for Indigenous Language Education" project.

It is largely based off the FastSpeech2/Fastpitch models.

## Quickstart

### Environment
We recommend using Conda and Python 3.9. To do that, create a new environment:

```sh
conda create --name EveryVoice python=3.9
conda activate EveryVoice
pip \
  install \
    torch==1.13.1+cu117 \
    torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117
```

### Install

Clone the repository and `pip` install it locally:

```sh
git clone https://github.com/roedoejet/EveryVoice.git
cd EveryVoice
git submodule update --init
pip install -e .
```

### Documentation

Read the full [EveryVoice documentation](https://docs.everyvoice.ca/).

In particular, read the [Guides](https://docs.everyvoice.ca/guides/index.html) to get familiar with the whole process.

## Contributing

Feel free to dive in! [Open an issue](https://github.com/roedoejet/EveryVoice/issues/new) or submit PRs.

This repo follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

You can install our standard Git hooks by running these commands in your sandbox:

```sh
pip install -r requirements.dev.txt
pre-commit install
gitlint install-hook
```

Have a look at [Contributing.md](Contributing.md) for the full details on the
Conventional Commit messages we prefer, our code formatting conventions, and
our Git hooks.

You can then interactively install the package by running the following command from the project root:

```sh
pip install -e .
```

## Acknowledgements

This repository draws on many other wonderful code bases.
Many thanks to:

https://github.com/nocotan/pytorch-lightning-gans
https://github.com/rishikksh20/iSTFTNet-pytorch
https://github.com/jik876/hifi-gan
https://github.com/ming024/FastSpeech2
https://github.com/MiniXC/LightningFastSpeech2
https://github.com/DigitalPhonetics/IMS-Toucan

## Tests

Run unit tests by `python -m unittest tests/test_configs.py` or suites of tests by running `everyvoice test dev` if you have the package installed interactively.
