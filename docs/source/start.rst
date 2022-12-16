.. _start:

Getting Started
================

Background
##########

There are approximately 70 Indigenous languages spoken in Canada from 10 distinct language families. As a consequence of the residential school system and other policies of cultural suppression, the majority of these languages now have fewer than 500 fluent speakers remaining, most of them elderly.

Despite this, Indigenous people have resisted colonial policies and continued speaking their languages, with interest by students and parents in Indigenous language education continuing to grow. Teachers are often overwhelmed by the number of students, and the trend towards online education means many students who have not previously had access to language classes now do. Supporting these growing cohorts of students comes with unique challenges in languages with few fluent first-language speakers. Teachers are particularly concerned with providing their students with opportunities to hear the language outside of class.

While there is no replacement for a speaker of an Indigenous language, there are possible applications for speech synthesis (text-to-speech) to supplement existing text-based tools like verb conjugators, dictionaries and phrasebooks.

The `National Research Council <https://nrc.canada.ca/en/research-development/research-collaboration/programs/canadian-indigenous-languages-technology-project>`_ has partnered with the `Onkwawenna Kentyohkwa Kanyen’kéha immersion school <https://onkwawenna.info/>`_, `W̱SÁNEĆ School Board <https://wsanecschoolboard.ca/>`_, `University nuhelot’įne thaiyots’į nistameyimâkanak Blue Quills <https://www.bluequills.ca/>`_, and the `University of Edinburgh <https://www.cstr.ed.ac.uk/>`_ to research and develop state-of-the-art speech synthesis (text-to-speech) systems and techniques for Indigenous languages in Canada, with a focus on how to integrate text-to-speech technology into the classroom.

The project is titled `Speech Generation for Indigenous Language Education <https://nrc.canada.ca/en/research-development/research-collaboration/programs/speech-generation-indigenous-language-education>`_ (SGILE).

More information and motivation for this project can be found in our `ACL2022 paper <https://aclanthology.org/2022.acl-long.507/>`_

What is |project| and this documentation about?
***********************************************

This project is the location of active research and development for the Speech Generation for Indigenous Language Education project. In addition to being a model for this project, it is meant to outline repeatable recipes for
other communities and languages to develop their own text-to-speech systems. This documentation describes guides for how to do this.


.. note::
   We are trying to develop a tool that makes the developer experience as smooth as possible. But, building these models and creating your datasets can be complicated.
   We recommend you are comfortable with `Python <https://try.codecademy.com/learn-python-3>`_ and using the `command line <https://missing.csail.mit.edu/2020/course-shell/>`_ before starting
   on this project.


Similar projects exist, why create another one?
***********************************************

It is true that similar excellent projects exist, such as `ESPnet <https://github.com/espnet/espnet>`_, `🐸TTS <https://github.com/coqui-ai/TTS>`_, `Comprehensive-Transformer-TTS <https://github.com/keonlee9420/Comprehensive-Transformer-TTS>`_, and `IMS-Toucan <https://github.com/DigitalPhonetics/IMS-Toucan>`_ among others.
Our reasons for creating our own are multi-fold (some of the following features are implemented in the aforementioned projects, but not every one of these features is supported in any of them):

- We support digraph/trigraph/multigraph inputs by defining our symbol sets as lists of strings instead of strings and using a custom tokenizer. This means we expect that your language might have inputs that consist of more than one symbol, like "kw" or "tl".
- We support out-of-the-box integration with `g2p <https://github.com/roedoejet/g2p>`_ which allows the g2p rules for 30+ Indigenous languages to be used in the project.
- We implement the use of multi-hot phonological feature vector inputs for easier pre-training/fine-tuning. We implement this using the `panphon library <https://github.com/dmort27/panphon>`_.
- We will not try to implement many different models. Instead we will curate a selection of models that we believe to be best for training models on under-resourced languages. In this way we are more similar to `IMS-Toucan <https://github.com/DigitalPhonetics/IMS-Toucan>`_ than `ESPnet <https://github.com/espnet/espnet>`_
- We use a custom, statically-typed configuration architecture between models written in `Pydantic <https://docs.pydantic.dev/>`_ that allows for configuration validation and serialization/de-serialization to json and yaml. It also allows us to ensure the same configuration for text and audio processing is used between models.
- We implement our models in `PyTorch Lightning <https://www.pytorchlightning.ai/>`_
- We implement separate and joint training of our feature prediction and vocoder models

.. note::
   These features do not necessarily mean that this is the right project for you. The other projects mentioned are of very high quality and might be a better fit for your project, particularly if you are lucky enough to have lots of data, or a langauge that is already supported.
