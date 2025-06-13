# EveryVoice TTS Toolkit 💬

[![codecov](https://codecov.io/gh/EveryVoiceTTS/EveryVoice/branch/main/graph/badge.svg?token=yErCxf64IU)](https://codecov.io/gh/EveryVoiceTTS/EveryVoice)
[![Documentation](https://github.com/EveryVoiceTTS/EveryVoice/actions/workflows/docs.yml/badge.svg)](https://docs.everyvoice.ca)
[![Build Status](https://github.com/EveryVoiceTTS/EveryVoice/actions/workflows/test.yml/badge.svg)](https://github.com/EveryVoiceTTS/EveryVoice/actions)
[![license](https://img.shields.io/badge/Licence-MIT-green)](LICENSE)
![alpha](https://img.shields.io/badge/alpha-grey)

This is the Text-to-Speech (TTS) toolkit used by the Small Teams "Speech Generation for Indigenous Language Education" project.

## Quickstart from PyPI

- Install Python 3.10, 3.11, or 3.12 and create a venv or a conda env for EveryVoice.

- Install `sox`.
  - On Ubuntu, `sudo apt-get install sox libsox-dev` should work.
  - Other Linux distros should have equivalent packages.
  - With Conda, `conda install sox -c conda-forge` is reliable.

- Install `ffmpeg`:
  - On Ubuntu, `sudo apt-get install ffmpeg` should work.
  - Other Linux distros should have an equivalent package.
  - With Conda, `conda install ffmpeg` is reliable.
  - Or, use the official bundles from https://www.ffmpeg.org/download.html

- Install `torch` and `torchaudio` version 2.1.0 for your platform and CUDA version: follow the instructions at https://pytorch.org/get-started/locally/ but specify `torch==2.1.0 torchaudio==2.1.0` in the install command and remove `torchvision`.

- Run `pip install everyvoice` (change the version to the current version if needed).

## Quickstart from source

### Install conda

First, you'll need to install `conda`. [Miniforge3](https://github.com/conda-forge/miniforge) is a fully open-source option which is free for all users and works well. You can also use Anaconda3 or Miniconda3 if you have or can get a license.

### Clone the repo

```sh
git clone https://github.com/EveryVoiceTTS/EveryVoice.git
cd EveryVoice
git submodule update --init
```

### Environment and installation – automated

To run EveryVoice, you need to create a new environment using Conda and Python 3.12, install all our dependencies and EveryVoice itself.

We have automated the procedure required to do all this in the script `make-everyvoice-env`, which you can run like this:

```sh
./make-everyvoice-env --path <env-path-of-your-choice>
conda activate <env-path-of-your-choice>
```

Add the option `--cuda CUDA_VERSION` if you need to override the default CUDA version, or `--cpu` to use Torch compiled for CPU use only.

### Environment and installation – manual

If the automated installation process does not work for you, or if you prefer to do the full installation manually, please refer to [EveryVoice / Installation](https://docs.everyvoice.ca/latest/install/#manual-installation).

### Documentation

Read the full [EveryVoice documentation](https://docs.everyvoice.ca/).

In particular, read the [Guides](https://docs.everyvoice.ca/latest/guides/) to get familiar with the whole process.

To build and view the documentation locally:
```
pip install -e '.[docs]'
mkdocs serve
```
and browse to http://127.0.0.1:8000/.

## Contributing

Feel free to dive in! [Open an issue](https://github.com/EveryVoiceTTS/EveryVoice/issues/new) or submit PRs.

This repo follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

Please make sure our standard Git hooks are activated, by running these commands in your sandbox (if you used our `make-everyvoice-env` script then this step is already done for you):

```sh
pip install -e '.[dev]'
pre-commit install
gitlint install-hook
git submodule foreach 'pre-commit install'
git submodule foreach 'gitlint install-hook'
```

Have a look at [Contributing.md](Contributing.md) for the full details on the
Conventional Commit messages we prefer, our code formatting conventions, our Git
hooks, and recommendations on how to make effective pull requests.

## Publishing Instructions

To publish a new version of the project, follow these steps:

1. **Determine the Version Bump**
   Decide whether your changes constitute a:
   - **Major** version bump (breaking changes),
   - **Minor** version bump (new features, backward-compatible, any change to the schema), or
   - **Patch** version bump (bug fixes, small changes).

2. **Update Version Files**
   - Update the `everyvoice._version` file to reflect the new version.
   - Keep all `submodule._version` files in sync with this version, **except** for the `wav2vec2` aligner submodule (which can be installed separately).
   - Commit the resulting changes, including all submodules.

3. **Update Schema (for Major/Minor bumps)**
   If you bumped a **major** or **minor** version:
   - Run `everyvoice update-schema`. You may need to delete existing schema files if you get an error message, but you should only do so if you are sure that those schema files have not already been published. I.e. we might already have schema files related to an alpha release - those can be overwritten, but we should never change published schema files.
   - Commit the resulting changes.

4. **Open a Pull Request**
   - Create a PR with your changes.
   - Wait for tests to pass and for the PR to be merged into `main`.

5. **Tag the Release**
   After merging:
   ```bash
   git tag -a -m vX.Y.Z vX.Y.Z
   git push 'vX.Y.Z'
   ```

6. **Update SchemaStore (for Major/Minor bumps)**
    Once the CI has built and released your version, if you bumped a major or minor version:

    Submit a PR to [SchemaStore](https://github.com/SchemaStore/schemastore) to update the schema reference.

    The only file you need to change is: `src/api/json/catalog.json`

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
