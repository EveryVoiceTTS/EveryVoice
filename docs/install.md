# Installation

In order to use EveryVoice on GPUs, you must install PyTorch and Cuda, Python 3.10 or more recent, a number of other dependencies, and EveryVoice itself. The following sections describe three ways to accomplish this:

## Scripted installation -- recommended

While the EveryVoice installation process has gotten simpler as the project matures,
we maintain a script to automate the process and keep it reliable.

 - Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [conda](https://docs.conda.io/projects/conda/en/stable/).
 - Clone the EveryVoice repo and its submodules:
    ```sh
	git clone https://github.com/EveryVoiceTTS/EveryVoice.git
	cd EveryVoice
	git submodule update --init
	```
 - Run our automated environment creation script
    ```sh
	./make-everyvoice-env --name EveryVoice
	conda activate EveryVoice
	```
	Add the option `--cuda CUDA_VERSION` if you need to override the default CUDA version, or `--cpu` to use Torch compiled for CPU use only.

## Install from PyPI

Direct installation from PyPI has become fairly reliable:

 - Follow the instructions in [README.md](https://github.com/EveryVoiceTTS/EveryVoice?tab=readme-ov-file) to install sox, ffmpeg, torch and torchaudio.

 - Install EveryVoice:

       pip install everyvoice==0.1.0a

## Manual installation using Conda

If you prefer to do the complete installation process manually, or if the automated process does not work for you, follow these steps.

### TL;DR

```
conda create --name EveryVoice python=3.12
conda activate EveryVoice
conda install sox -c conda-forge
conda install ffmpeg
CUDA_TAG=cu118 pip install -r requirements.torch.txt --find-links https://download.pytorch.org/whl/torch_stable.html
pip install -e .
```

### Install Conda

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [conda](https://docs.conda.io/projects/conda/en/stable/).

### Create the environment

Use conda to create a new environment based on Python 3.12:
```sh
conda create --name EveryVoice python=3.12
conda activate EveryVoice
```

### Pytorch dependencies

Install our pytorch requirements from `requirements.torch.txt`, replacing `cu118` below (for
CUDA 11.8) by your actual CUDA version tag (118 or higher), or by `cpu` for a CPU-only installation:

```sh
CUDA_TAG=cu118 pip install -r requirements.torch.txt --find-links https://download.pytorch.org/whl/torch_stable.html
```

Alternatively, you can follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) relevant to your hardware.
Make sure you specify the version declared in `requirements.torch.txt`, which is 2.3.1 at the moment
if you install EveryVoice from GitHub, but 2.1.0 if you install it from PyPI.

### Non-Python dependencies

Install sox and ffmpeg if you didn't install them using OS packages:
```sh
conda install sox -c conda-forge
conda install ffmpeg
```

### Handling running out of temp disk space

Installation will require a fair bit of space on `~/.cache` and your `$TMPDIR`
(`/tmp` by default, if `$TMPDIR` is not set).  If you get the error
`OSError: [Errno 28] No space left on device` during installation, you may need
to do one or both of these operations:
 - `export TMPDIR=/path/to/a/large/tmp/space` (or maybe `export TMPDIR=.`)
 - `mkdir /path/to/a/large/filesystem/.cache; ln -s /path/to/a/large/filesystem/.cache ~/.cache`

### Install EveryVoice itself

Install EveryVoice locally from your cloned sandbox:

```sh
pip install -e .
```

### Dev dependencies

Before you can run the test suites, you'll also need to install the dev dependencies:

```sh
pip install -e .[dev]
```

### Git hooks

If you plan to contribute to the project, please install our Git hooks:

```sh
pre-commit install
gitlint install-hook
git submodule foreach 'pre-commit install'
git submodule foreach 'gitlint install-hook'
```

## Installation using `uv`

If you can install sox, libsox-dev and ffmpeg using your OS packages or by other means (see [README.md](https://github.com/EveryVoiceTTS/EveryVoice?tab=readme-ov-file#quickstart-from-pypi)),
you can now install EveryVoice in a [`uv`](https://docs.astral.sh/uv/) venv, which is much faster to create and activate.

```
uv venv -p 3.12 .venv-EveryVoice
source .venv-EveryVoice/bin/activate
uv pip install torch==2.3.1+cu118 torchaudio==2.3.1+cu118 --find-links https://download.pytorch.org/whl/torch_stable.html
uv pip install -e .[dev]
```

(If needed, change the `+cu118` qualifier on torch\* to your actual version of CUDA, or to `+cpu`.
See [Pytorch dependencies](#pytorch-dependencies) above.)
