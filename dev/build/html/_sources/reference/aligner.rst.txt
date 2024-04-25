.. _aligner:

Aligner
=============

Source Code: https://github.com/roedoejet/DeepForcedAligner

The Aligner module is for training a model of alignment between your data's text and speech. In other implementations, this is handled by a 3rd party aligner, like the `Montreal Forced Aligner (MFA) <https://montreal-forced-aligner.readthedocs.io/en/latest/>`_.
Similar to `IMS-Toucan <https://github.com/DigitalPhonetics/IMS-Toucan>`_, we have implemented our own based on `DeepForcedAligner <https://github.com/as-ideas/DeepForcedAligner>`_ to ensure that the way our model processes text and audio remains consistent between
the aligner and the other models.


Command Line Interface
----------------------

.. click:: everyvoice.cli:CLICK_APP
    :prog: everyvoice
    :nested: full
    :commands: dfa

Configuration
-------------

Main Configuration
******************

.. autopydantic_settings:: everyvoice.model.aligner.DeepForcedAligner.dfaligner.config.DFAlignerConfig
    :settings-show-json: True
    :settings-show-config-member: False
    :settings-show-config-summary: False
    :settings-show-validator-members: True
    :settings-show-validator-summary: True
    :field-list-validators: True

Training Configuration
**********************

.. autopydantic_settings:: everyvoice.model.aligner.DeepForcedAligner.dfaligner.config.DFAlignerTrainingConfig
    :settings-show-json: True
    :settings-show-config-member: False
    :settings-show-config-summary: False
    :settings-show-validator-members: True
    :settings-show-validator-summary: True
    :field-list-validators: True

Model Configuration
*******************

.. autopydantic_settings:: everyvoice.model.aligner.DeepForcedAligner.dfaligner.config.DFAlignerModelConfig
    :settings-show-json: True
    :settings-show-config-member: False
    :settings-show-config-summary: False
    :settings-show-validator-members: True
    :settings-show-validator-summary: True
    :field-list-validators: True
