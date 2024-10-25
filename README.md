:construction: :construction: Warning! This repository is not quite ready yet. We're releasing it publicly in alpha, but it should be expected to change drastically over the coming months. :construction: :construction:

# EveryVoice TTS Toolkit 💬

[![codecov](https://codecov.io/gh/EveryVoiceTTS/EveryVoice/branch/main/graph/badge.svg?token=yErCxf64IU)](https://codecov.io/gh/EveryVoiceTTS/EveryVoice)
[![Documentation](https://github.com/EveryVoiceTTS/EveryVoice/actions/workflows/docs.yml/badge.svg)](https://docs.everyvoice.ca)
[![Build Status](https://github.com/EveryVoiceTTS/EveryVoice/actions/workflows/test.yml/badge.svg)](https://github.com/EveryVoiceTTS/EveryVoice/actions)
[![license](https://img.shields.io/badge/Licence-MIT-green)](LICENSE)

This is the Text-to-Speech (TTS) toolkit used by the Small Teams "Speech Generation for Indigenous Language Education" project.

## Quickstart

### Install conda

First, you'll need to install `conda`. [Miniforge3](https://github.com/conda-forge/miniforge) is a fully open-source option which is free for all users and works well. You can also use Anaconda3 or Miniconda3 if you have or can get a license.

### Clone the repo

```sh
git clone https://github.com/EveryVoiceTTS/EveryVoice.git
cd EveryVoice
git submodule update --init
```

### Environment and installation – automated

To run EveryVoice, you need to create a new environment using Conda and Python 3.10, install all our dependencies and EveryVoice itself.

We have automated the procedure required to do all this in the script `make-everyvoice-env`, which you can run like this:

```sh
./make-everyvoice-env --name <env-name-of-your-choice>
conda activate <env-name-of-your-choice>
```

Add the option `--cuda CUDA_VERSION` if you need to override the default CUDA version, or `--cpu` to use Torch compiled for CPU use only.

### Environment and installation – manual

If the automated installation process does not work for you, or if you prefer to do the full installation manually, please refer to [EveryVoice / Installation](https://docs.everyvoice.ca/latest/install/#manual-installation).

### Documentation

Read the full [EveryVoice documentation](https://docs.everyvoice.ca/).

In particular, read the [Guides](https://docs.everyvoice.ca/latest/guides/) to get familiar with the whole process.

## Contributing

Feel free to dive in! [Open an issue](https://github.com/EveryVoiceTTS/EveryVoice/issues/new) or submit PRs.

This repo follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

Please make sure our standard Git hooks are activated, by running these commands in your sandbox (if you used our `make-everyvoice-env` script then this step is already done for you):

```sh
pip install -r requirements.dev.txt
pre-commit install
gitlint install-hook
git submodule foreach 'pre-commit install'
git submodule foreach 'gitlint install-hook'
```

Have a look at [Contributing.md](Contributing.md) for the full details on the
Conventional Commit messages we prefer, our code formatting conventions, our Git
hooks, and recommendations on how to make effective pull requests.

## Acknowledgements

This repository draws on many other wonderful code bases.
Many thanks to:

- https://github.com/nocotan/pytorch-lightning-gans
- https://github.com/rishikksh20/iSTFTNet-pytorch
- https://github.com/jik876/hifi-gan
- https://github.com/ming024/FastSpeech2
- https://github.com/MiniXC/LightningFastSpeech2
- https://github.com/DigitalPhonetics/IMS-Toucan

## Tests

Run unit tests by `python -m unittest tests/test_configs.py` or suites of tests by running `everyvoice test dev` if you have the package installed interactively.
