# Configuration

Each model has a statically typed configuration model. Each configuration has default settings that will be instantiated when the model is instantiated. To create a default preprocessing configuration for example you would:

```python
from everyvoice.config.preprocessing_config import PreprocessingConfig

preprocessing_config = PreprocessingConfig()
```

Static typing means that if someone accidentally enters an integer that the configuration expects should be a float or some other trivial typing difference, it is
corrected when the configuration is instantiated, and doesn't produce downstream runtime errors. It also means that intellisense is available in your code editor
when working with a configuration class.

## New Dataset Wizard 🧙‍♀️

This section of the documentation is meant for technical explanations and reference documentation - if you're just looking to get started, have a look at the {ref}`guides` section instead to learn about how the New Dataset Wizard 🧙‍♀️ can help you configure everything for your dataset.

Here is the reference documentation for the New Dataset Wizard 🧙‍♀️

<!-- ::: mkdocs-typer
    :prog_name: everyvoice
    :module: everyvoice.cli.app
    :command: new-dataset -->

<!-- ::: mkdocs-click
    :prog_name: everyvoice
    :module: everyvoice.cli
    :command: CLICK_APP -->

<!-- ```{eval-rst}
.. click:: everyvoice.cli:CLICK_APP
    :prog: everyvoice
    :nested: full
    :commands: new-dataset

``` -->

## Sharing Configurations

The Text and Preprocessing configurations should only be defined once per dataset and shared between your models to ensure each model makes the same assumptions about your data.
To achieve that, each model configuration can also be defined as a path to a configuration file. So, a configuration for an [aligner model](./aligner.md) that uses separately defined text and audio preprocessing configurations might look like this:

```yaml hl_lines="8 9"

model:
    lstm_dim: 512
    conv_dim: 512
    ...
training:
    batch_size: 32
    ...
preprocessing: "./config/default/{{ config_filename('preprocessing') }}"
text: "./config/default/{{ config_filename('text') }}"
```

## Serialization

By default configuration objects are serialized as dictionaries, which works as expected with integers, floats, lists, booleans, dicts etc. But there are some cases where you need to specify a Callable in your configuration. For example the {ref}`TextConfig` has a `cleaners` field that takes a list of Callables to apply in order to raw text.
By default, these functions turn raw text to lowercase, collapse whitespace, and normalize using Unicode NFC normalization. In Python, we could instantiate this by passing the callables directly like so:

```python


from everyvoice.config.text_config import TextConfig
from everyvoice.utils import collapse_whitespace, lower, nfc_normalize

text_config = TextConfig(cleaners=[lower, collapse_whitespace, nfc_normalize])
```

But, for yaml or json configuration, we need to serialize these functions. To do so, EveryVoice will turn each callable into module dot-notation. That is,
your configuration will look like this in yaml:

```yaml
cleaners:
    - everyvoice.utils.lower
    - everyvoice.utils.collapse_whitespace
    - everyvoice.utils.nfc_normalize
```

This will then be de-serialized upon instantiation of your configuration.

## Text Configuration

The TextConfig is where you define the symbol set for your data and any cleaners used to clean your raw text into the text needed
for your data. You can share the TextConfig with any models that need it and only need one text configuration per dataset (and possibly only per language).

!!! note
    Only the [aligner](./aligner.md), [feature_prediction](./feature_prediction.md), and [e2e](./e2e.md) models need text configuration. The [vocoder](./vocoder.md) is trained without text and does not need a TextConfig. For more information on this, see the [background](../guides/background.md) section.


### TextConfig

::: everyvoice.config.text_config.TextConfig
    handler: python
    options:
        members:
            - cleaners
            - symbols
            - to_replace
        show_root_heading: true
        heading_level: 6

<!-- ```{eval-rst}
.. autopydantic_settings:: everyvoice.config.text_config.TextConfig
    :settings-show-json: False
    :settings-show-config-member: False
    :settings-show-config-summary: False
    :settings-show-validator-members: True
    :settings-show-validator-summary: True
    :field-list-validators: True
``` -->

### Symbols

```{eval-rst}
.. autopydantic_settings:: everyvoice.config.text_config.Symbols
    :settings-show-json: True
    :settings-show-config-member: False
    :settings-show-config-summary: False
    :settings-show-validator-members: True
    :settings-show-validator-summary: True
    :field-list-validators: True

```

## Preprocessing Configuration
