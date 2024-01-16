# Installation

In order to train on GPUs, you must install PyTorch and Cuda. To do so, we recommend:

- installing [conda](https://docs.conda.io/projects/conda/en/stable/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- creating a new environment: `conda create --name EveryVoice python=3.10`
- activating the environment: `conda activate EveryVoice`
- following the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) relevant to your hardware

We then recommend using an interactive installation after cloning the repo from GitHub:

```bash
$ git clone https://github.com/roedoejet/EveryVoice.git
$ cd EveryVoice
$ git submodule update --init
$ conda activate EveryVoice
$ pip install -e .
```
