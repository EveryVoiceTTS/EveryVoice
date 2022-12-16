.. _vocoder:

Vocoder
=============

HiFiGAN/iSTFT-Net
-----------------

Source Code: https://github.com/roedoejet/HiFiGAN_iSTFT_lightning

This vocoder is based on the HiFiGAN/iSTFT-Net neural vocoders.

Command Line Interface
**********************

.. click:: smts.cli:CLICK_APP
    :prog: smts
    :nested: full
    :commands: hifigan

Configuration
*************

Main Configuration
~~~~~~~~~~~~~~~~~~

.. autopydantic_settings:: smts.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config.HiFiGANConfig
    :settings-show-json: True
    :settings-show-config-member: False
    :settings-show-config-summary: False
    :settings-show-validator-members: True
    :settings-show-validator-summary: True
    :field-list-validators: True

Training Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. autopydantic_settings:: smts.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config.HiFiGANTrainingConfig
    :settings-show-json: True
    :settings-show-config-member: False
    :settings-show-config-summary: False
    :settings-show-validator-members: True
    :settings-show-validator-summary: True
    :field-list-validators: True

Model Configuration
~~~~~~~~~~~~~~~~~~~

.. autopydantic_settings:: smts.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config.HiFiGANModelConfig
    :settings-show-json: True
    :settings-show-config-member: False
    :settings-show-config-summary: False
    :settings-show-validator-members: True
    :settings-show-validator-summary: True
    :field-list-validators: True
