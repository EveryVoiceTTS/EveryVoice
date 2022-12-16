.. _install:

Installation
=============

In order to train on GPUs, you must install PyTorch and Cuda. To do so, we recommend:

- installing `conda <https://docs.conda.io/projects/conda/en/stable/>`_ or `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
- creating a new environment: ``conda create --name SmallTeamSpeech python=3.9``
- activating the environment: ``conda activate SmallTeamSpeech``
- following the `PyTorch installation instructions <https://pytorch.org/get-started/locally/>`_ relevant to your hardware

We then recommend using an interactive installation after cloning the repo from GitHub:

.. code-block:: bash

    $ git clone https://github.com/roedoejet/SmallTeamSpeech.git
    $ cd SmallTeamSpeech
    $ git submodule update --init
    $ conda activate SmallTeamSpeech
    $ pip install -e .
