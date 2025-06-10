# Installation

In order to use EveryVoice on GPUs, you must install PyTorch and Cuda, Python 3.10 or more recent, a number of other dependencies, and EveryVoice itself. The following sections describe various ways to accomplish this.

## Installation from PyPI

Direct installation from PyPI is fairly reliable.

 - Follow the instructions in [README.md](https://github.com/EveryVoiceTTS/EveryVoice?tab=readme-ov-file) to install sox, ffmpeg, torch and torchaudio.

 - Install EveryVoice:

=== "Using conda"

        pip install everyvoice==0.2.0a1

=== "Using uv"

        uv pip install everyvoice==0.2.0a1

## Installation from source

### Scripted installation &mdash; recommended

While the EveryVoice installation process has gotten simpler as the project matures,
we maintain a script to automate the process and keep it reliable.

=== "Using conda"

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
		Add the option `--cuda CUDA_VERSION` if you need to override the default CUDA version,
		or `--cpu` to use Torch compiled for CPU use only.

=== "Using uv"

	If you can install sox, libsox-dev and ffmpeg using your OS packages or by other means
	(see [README.md](https://github.com/EveryVoiceTTS/EveryVoice?tab=readme-ov-file#quickstart-from-pypi)),
	you can install EveryVoice in a [`uv`](https://docs.astral.sh/uv/) venv; otherwise, use conda.

    - [Install uv](https://docs.astral.sh/uv/getting-started/installation/).
	- Clone the EveryVoice repo and its submodules:
		```sh
		git clone https://github.com/EveryVoiceTTS/EveryVoice.git
		cd EveryVoice
		git submodule update --init
		```
	- Run our automated environment creation script
		```sh
		./make-everyvoice-env --uv
		source .venv/bin/activate
		```
		Add the option `--cuda CUDA_VERSION` if you need to override the default CUDA version,
	    or `--cpu` to use Torch compiled for CPU use only.

### Manual installation &mdash; TL;DR

=== "Using conda"

	```
	conda create --name EveryVoice python=3.12 ffmpeg
	conda activate EveryVoice
	conda install sox -c conda-forge
	CUDA_TAG=cu121 pip install -r requirements.torch.txt --find-links https://download.pytorch.org/whl/torch_stable.html
	pip install -e .[dev]
	```

=== "Using uv"

    First install [sox, libsox-dev and ffmpeg](https://github.com/EveryVoiceTTS/EveryVoice?tab=readme-ov-file#quickstart-from-pypi) externally.

	```
	uv venv -p 3.12 .venv
	source .venv/bin/activate
	uv pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 --find-links https://download.pytorch.org/whl/torch_stable.html
	uv pip install -e .[dev]
	```

### Manual installation &mdash; Detailed

If you prefer to do the complete installation process manually, or if the
automated process does not work for you, follow these steps.

#### Install your package manager

=== "Using conda"

	Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [conda](https://docs.conda.io/projects/conda/en/stable/).

=== "Using uv"

    [Install uv](https://docs.astral.sh/uv/getting-started/installation/).

#### Create the environment

=== "Using conda"

	Use conda to create a new environment based on Python 3.12:
	```sh
	conda create --name EveryVoice python=3.12
	conda activate EveryVoice
	```

=== "Using uv"

    Use uv to create a new environment based on Python 3.12:
	```sh
	uv venv -p 3.12
	source .venv/bin/activate
	```

#### Pytorch dependencies

=== "Using conda"

	Install our pytorch requirements from `requirements.torch.txt`, replacing `cu121` below (for
	CUDA 12.1) by your actual CUDA version tag (cu118 or higher), or by `cpu` for a CPU-only installation:

	```sh
	CUDA_TAG=cu121 pip install -r requirements.torch.txt --find-links https://download.pytorch.org/whl/torch_stable.html
	```

=== "Using uv"

    Install our pytorch requirements specified in `requirements.torch.txt`, but manually.
	(Unfortunately, `uv` does not support the environment variable we use with pip and conda.)
	Replace `cu121` by your actual CUDA version tag, or by `cpu` for a CPU-only installation.

	```sh
	uv pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 --find-links https://download.pytorch.org/whl/torch_stable.html
	```

Alternatively, you can follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) relevant to your hardware.
Make sure you specify the version declared in `requirements.torch.txt`, which is 2.3.1 at the moment.
if you install EveryVoice from GitHub, but 2.1.0 if you install it from PyPI.

#### Non-Python dependencies

=== "Using conda"

	Install sox and ffmpeg with conda if you didn't install them using OS packages:
	```sh
	conda install sox -c conda-forge
	conda install ffmpeg
	```

=== "Using uv"

    Uv does not provide a mechanism for installing non-Python dependencies.
	You must follow the sox, libsox-dev and ffmpeg installation instructions in our
	[README.md](https://github.com/EveryVoiceTTS/EveryVoice?tab=readme-ov-file#quickstart-from-pypi)
	if you want to use uv.

#### Handling running out of temp disk space

Installation will require a fair bit of space on `~/.cache` and/or `~/.local`, as well as your `$TMPDIR`
(`/tmp` by default, if `$TMPDIR` is not set).  If you get the error
`OSError: [Errno 28] No space left on device` during installation, you may need
to do one or more of these operations:

 - `export TMPDIR=/path/to/a/large/tmp/space` (or maybe `export TMPDIR=.`)
 - `mkdir /path/to/a/large/filesystem/.cache; ln -s /path/to/a/large/filesystem/.cache ~/.cache`
 - `mkdir /path/to/a/large/filesystem/.local; ln -s /path/to/a/large/filesystem/.local ~/.local`

#### Install EveryVoice itself

Install EveryVoice locally from your cloned sandbox:

=== "Using conda"

	```sh
	pip install -e .
	```

=== "Using uv"

	```sh
	uv pip install -e .
	```

#### Dev dependencies

Before you can run the test suites, you'll also need to install the dev dependencies:

=== "Using conda"

	```sh
	pip install -e .[dev]
	```

=== "Using uv"

	```sh
	uv pip install -e .[dev]
	```

#### Git hooks

If you plan to contribute to the project, please install our Git hooks:

```sh
pre-commit install
gitlint install-hook
git submodule foreach 'pre-commit install'
git submodule foreach 'gitlint install-hook'
```
