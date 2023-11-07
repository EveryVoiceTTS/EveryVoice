# EveryVoice TTS Toolkit 💬

[![codecov](https://codecov.io/gh/roedoejet/EveryVoice/branch/main/graph/badge.svg?token=yErCxf64IU)](https://codecov.io/gh/roedoejet/EveryVoice)
[![license](https://img.shields.io/badge/Licence-MIT-green)](LICENSE)

This is the Text-to-Speech (TTS) toolkit used by the Small Teams "Speech Generation for Indigenous Language Education" project.

## Quickstart

### Environment
We recommend using Conda and Python 3.9. To do that, create a new environment,
replacing cu117 (for CUDA 11.7) by your actual CUDA version's tag. These steps have been automated and can be triggered with:

```sh
git clone --recursive https://github.com/roedoejet/EveryVoice.git
cd EveryVoice
bash make-fresh-env.sh <env-name-of-your-choice>
conda activate <env-name-of-your-choice>
```

Installation will require a fair bit of space on `~/.cache` and your `$TMPDIR`
(`/tmp` by default, if `$TMPDIR` is not set).  If you get the error
`OSError: [Errno 28] No space left on device` during installation, you may need
to do one or both of these operations:
 - `export TMPDIR=/path/to/a/large/tmp/space` (or maybe `export TMPDIR=.`)
 - `mkdir /path/to/a/large/filesystem/.cache; ln -s /path/to/a/large/filesystem/.cache ~/.cache`

### Documentation

Read the full [EveryVoice documentation](https://docs.everyvoice.ca/).

In particular, read the [Guides](https://docs.everyvoice.ca/guides/index.html) to get familiar with the whole process.

## Contributing

Feel free to dive in! [Open an issue](https://github.com/roedoejet/EveryVoice/issues/new) or submit PRs.

This repo follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

Make sure our standard Git hooks are running these commands in your sandbox (if you used our `make-fresh-env.sh` script then this step is already done for you.):

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

https://github.com/nocotan/pytorch-lightning-gans
https://github.com/rishikksh20/iSTFTNet-pytorch
https://github.com/jik876/hifi-gan
https://github.com/ming024/FastSpeech2
https://github.com/MiniXC/LightningFastSpeech2
https://github.com/DigitalPhonetics/IMS-Toucan

## Tests

Run unit tests by `python -m unittest tests/test_configs.py` or suites of tests by running `everyvoice test dev` if you have the package installed interactively.
