# EveryVoice TTS Toolkit 💬

[![codecov](https://codecov.io/gh/roedoejet/EveryVoice/branch/main/graph/badge.svg?token=yErCxf64IU)](https://codecov.io/gh/roedoejet/EveryVoice)
[![license](https://img.shields.io/badge/Licence-MIT-green)](LICENSE)

This is the Text-to-Speech (TTS) toolkit used by the Small Teams "Speech Generation for Indigenous Language Education" project.

## Quickstart

### Clone the repo

```sh
git clone https://github.com/roedoejet/EveryVoice.git
cd EveryVoice
git submodule update --init
```

### Environment and installation – automated

To run EveryVoice, you need to create a new environment using Conda and Python 3.9, install all our dependencies and EveryVoice itself.

We have automated the procedure required to do all this in the script `make-fresh-env.sh`, which you can run like this:

```sh
bash make-fresh-env.sh <env-name-of-your-choice>
conda activate <env-name-of-your-choice>
```

### Environment and installation – manual

#### Create the environment

Use conda to create a new environment based on Python 3.9, replacing cu118 (for
CUDA 11.8) by your actual CUDA version tag (118 or higher):

```sh
conda create --name EveryVoice python=3.9
conda activate EveryVoice
CUDA_TAG=cu118 pip install -r requirements.torch.txt --find-links https://download.pytorch.org/whl/torch_stable.html
pip install cython
```

Installation will require a fair bit of space on `~/.cache` and your `$TMPDIR`
(`/tmp` by default, if `$TMPDIR` is not set).  If you get the error
`OSError: [Errno 28] No space left on device` during installation, you may need
to do one or both of these operations:
 - `export TMPDIR=/path/to/a/large/tmp/space` (or maybe `export TMPDIR=.`)
 - `mkdir /path/to/a/large/filesystem/.cache; ln -s /path/to/a/large/filesystem/.cache ~/.cache`

#### Install

Install EveryVoice locally from your cloned sandbox:

```sh
pip install -e .
```

### Installation for use on CPU only

The steps above use CUDA_VERSION (automated) or CUDA_TAG (manual) to specify the version of CUDA you are using. If you want to train a system for use on CPU only, use the value `cpu`:

```sh
CUDA_VERSION=cpu bash make-fresh-env.sh EveryVoice-cpu
```

or `CUDA_TAG=cpu` in the manual installation process.

### Documentation

Read the full [EveryVoice documentation](https://docs.everyvoice.ca/).

In particular, read the [Guides](https://docs.everyvoice.ca/guides/index.html) to get familiar with the whole process.

## Contributing

Feel free to dive in! [Open an issue](https://github.com/roedoejet/EveryVoice/issues/new) or submit PRs.

This repo follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

Please make sure our standard Git hooks are activated, by running these commands in your sandbox (if you used our `make-fresh-env.sh` script then this step is already done for you.):

```sh
pip install -r requirements.dev.txt
pre-commit install
gitlint install-hook
```

Have a look at [Contributing.md](Contributing.md) for the full details on the
Conventional Commit messages we prefer, our code formatting conventions, and
our Git hooks.

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
