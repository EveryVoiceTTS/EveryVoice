.. _e2e:

E2E
=============


Command Line Interface
-----------------------

.. click:: smts.cli:CLICK_APP
    :prog: smts
    :nested: full
    :commands: e2e


Configuration
*************

Main Configuration
~~~~~~~~~~~~~~~~~~

.. autopydantic_settings:: smts.model.e2e.config.SMTSConfig
    :settings-show-json: True
    :settings-show-config-member: False
    :settings-show-config-summary: False
    :settings-show-validator-members: True
    :settings-show-validator-summary: True
    :field-list-validators: True
