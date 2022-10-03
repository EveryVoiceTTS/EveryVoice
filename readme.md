# Small Team Speech Model

[![codecov](https://codecov.io/gh/roedoejet/SmallTeamSpeech/branch/main/graph/badge.svg?token=yErCxf64IU)](https://codecov.io/gh/roedoejet/SmallTeamSpeech)
[![license](https://img.shields.io/badge/Licence-MIT-green)](LICENSE)

This is an implementation of the Text-to-Speech (TTS) model used by the Small Teams "Speech Generation for Indigenous Language Education" project.

It is largely based off the FastSpeech2/Fastpitch models.

## Quickstart

### Dependencies

I recommend using Conda and Python 3.9. To do that, create a new environment:

```
conda create --name SmallTeamSpeech python=3.9
conda activate SmallTeamSpeech
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

You can then install the rest of the Python dependencies with pip

```
pip3 install -r requirements.txt
```

Alternatively, you can just create the whole thing from our `environment.yml` file:

```
conda env create -f environment.yml
conda activate SmallTeamSpeech
```

## Contributing

Feel free to dive in! [Open an issue](https://github.com/roedoejet/SmallTeamSpeech/issues/new) or submit PRs.

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

## Tests

Run unit tests by `python3 -m unittest tests/test_configs.py`
