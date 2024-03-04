# Installation

In order to use EveryVoice on GPUs, you must install PyTorch and Cuda, Python 3.10 or more recent, a number of other dependencies, and EveryVoice itself. The following sections describe three ways to accomplish this:

## Using Pip

We hope direct installation from PyPI will work:

 - Follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) relevant to your hardware.

 - Install EveryVoice:

       pip install everyvoice

## Scripted installation -- recommended

The EveryVoice installation process can be somewhat involved, so we have automated it as much as we could.

 - Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [conda](https://docs.conda.io/projects/conda/en/stable/).
 - Clone the EveryVoice repo and its submodules:
    ```sh
	git clone https://github.com/roedoejet/EveryVoice.git
	cd EveryVoice
	git submodule update --init
	```
 - Run our automated environment creation script
    ```sh
	./make-everyvoice-env --name EveryVoice
	conda activate EveryVoice
	```
	Add the option `--cuda CUDA_VERSION` if you need to override the default CUDA version, or `--cpu` to use Torch compiled for CPU use only.

## Manual installation

If you prefer to do the complete installation process manually, or if the automated process does not work for you, follow these steps:

### Install Conda

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [conda](https://docs.conda.io/projects/conda/en/stable/).


### Create the environment

Use conda to create a new environment based on Python 3.10:
```sh
conda create --name EveryVoice python=3.10
conda activate EveryVoice
```

### Pytorch dependencies

Install our pytorch requirements from `requirements.torch.txt`, replacing `cu118` below (for
CUDA 11.8) by your actual CUDA version tag (118 or higher), or by `cpu` for a CPU-only installation:

```sh
CUDA_TAG=cu118 pip install -r requirements.torch.txt --find-links https://download.pytorch.org/whl/torch_stable.html
```

Alternatively, you can follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) relevant to your hardware.

### Other potentially tricky dependencies

These requirements sometimes require being run separately:
```sh
pip install cython
conda install sox -c conda-forge
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
pip install -r requirements.dev.txt
```

### Git hooks

If you plan to contribute to the project, please install our Git hooks:

```sh
pre-commit install
gitlint install-hook
git submodule foreach 'pre-commit install'
git submodule foreach 'gitlint install-hook'
```
