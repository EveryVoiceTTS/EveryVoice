# Getting Started

Welcome to the EveryVoice TTS Toolkit Documentation page! Please read the background section below to hear a bit about how this project got started, or head over to the [guides](./guides/index.md) section to find a guide to help you get started.

## Background

There are approximately 70 Indigenous languages spoken in Canada from 10 distinct language families. As a consequence of the residential school system and other policies of cultural suppression, the majority of these languages now have fewer than 500 fluent speakers remaining, most of them elderly.

Despite this, Indigenous people have resisted colonial policies and continued speaking their languages, with interest by students and parents in Indigenous language education continuing to grow. Teachers are often overwhelmed by the number of students, and the trend towards online education means many students who have not previously had access to language classes now do. Supporting these growing cohorts of students comes with unique challenges in languages with few fluent first-language speakers. Teachers are particularly concerned with providing their students with opportunities to hear the language outside of class.

While there is no replacement for a speaker of an Indigenous language, there are possible applications for speech synthesis (text-to-speech) to supplement existing text-based tools like verb conjugators, dictionaries and phrasebooks.

The [National Research Council](https://nrc.canada.ca/en/research-development/research-collaboration/programs/canadian-indigenous-languages-technology-project) has partnered with the [Onkwawenna Kentyohkwa Kanyen’kéha immersion school](https://onkwawenna.info/), [W̱SÁNEĆ School Board](https://wsanecschoolboard.ca/), [University nuhelot’įne thaiyots’į nistameyimâkanak Blue Quills](https://www.bluequills.ca/), and the [University of Edinburgh](https://www.cstr.ed.ac.uk/) to research and develop state-of-the-art speech synthesis (text-to-speech) systems and techniques for Indigenous languages in Canada, with a focus on how to integrate text-to-speech technology into the classroom.

The project is titled [Speech Generation for Indigenous Language Education](https://nrc.canada.ca/en/research-development/research-collaboration/programs/speech-generation-indigenous-language-education) (SGILE) and the EveryVoice TTS toolkit is one of the products of this collaboration. Detailed information about this project can be found in our [recent submission](assets/sgile_preprint.pdf) to Computer Speech & Language (Under Review).

### What is EveryVoice and this documentation about?

This project is the location of active research and development for the Speech Generation for Indigenous Language Education project. In addition to being a model for this project, it is meant to outline repeatable recipes for
other communities and languages to develop their own text-to-speech systems. This documentation describes guides for how to do this.

!!! note
    We are trying to develop a tool that makes the developer experience as smooth as possible. But, building these models and creating your datasets can be complicated.
    We recommend you are comfortable with [Python](https://try.codecademy.com/learn-python-3) and using the [command line](https://missing.csail.mit.edu/2020/course-shell/) before starting
    on this project.


### Similar projects exist, why create another one?

It is true that similar excellent projects exist, such as [ESPnet](https://github.com/espnet/espnet), [🐸TTS](https://github.com/coqui-ai/TTS), [Comprehensive-Transformer-TTS](https://github.com/keonlee9420/Comprehensive-Transformer-TTS), and [IMS-Toucan](https://github.com/DigitalPhonetics/IMS-Toucan) among others.
Our reasons for creating our own are multi-fold (some of the following features are implemented in the aforementioned projects, but not every one of these features is supported in any of them):

- The EveryVoice TTS toolkit comes with a configuration wizard which helps configure the toolkit to new datasets in new languages.
- We support a heterogeneous source of data, meaning you (hopefully) have to do less work to wrangle data together. The configuration wizard supports multi-dataset configuration.
- We support out-of-the-box integration with [g2p](https://github.com/roedoejet/g2p) which allows the g2p rules for 30+ Indigenous languages to be used in the project.
- We will not try to implement many different models. Instead we will curate a model architecture that we believe to be best for training models on under-resourced languages. In this way we are more similar to [IMS-Toucan](https://github.com/DigitalPhonetics/IMS-Toucan) than [ESPnet](https://github.com/espnet/espnet)
- We use a custom, statically-typed configuration architecture between models written in [Pydantic](https://docs.pydantic.dev/) that allows for configuration validation and serialization/de-serialization to json and yaml. It also allows us to ensure the same configuration for text and audio processing is used between models.
- We implement our models in [PyTorch Lightning](https://www.pytorchlightning.ai/)

For a detailed comparison of selected features of EveryVoice and other toolkits please see Appendix B & C in our [recent paper submission](assets/sgile_preprint.pdf).

!!! note
    These features do not necessarily mean that this is the right project for you. The other projects mentioned are of very high quality and might be a better fit for your project, particularly if you are lucky enough to have lots of data, or a language that is already supported.
