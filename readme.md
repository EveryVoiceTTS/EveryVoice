# Small Team Speech Model

This is an implementation of the Text-to-Speech (TTS) model used by the Small Teams "Speech Generation for Indigenous Language Education" project.

It is largely based off the FastSpeech2/Fastpitch models.

## Quickstart

### Dependencies

I recommend using Conda and Python 3.7. To do that, create a new environment:

```
conda create --name SmallTeamsSpeech python=3.7
conda activate SmallTeamsSpeech
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

You can then install the Python dependencies with pip or conda

```
pip3 install -r requirements.txt
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

## Tests

Run unit tests by `python3 -m unittest tests/test_configs.py`
