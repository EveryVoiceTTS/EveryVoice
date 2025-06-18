# Configuration

Each model has a statically typed configuration model. Each configuration has default settings that will be instantiated when the model is instantiated. To create a default preprocessing configuration for example you would:

```python
from everyvoice.config.preprocessing_config import PreprocessingConfig

preprocessing_config = PreprocessingConfig()
```

Static typing means that misconfiguration errors should occur as soon as the configuration is instantiated instead of producing downstream runtime errors. It also means that intellisense is available in your code editor when working with a configuration class.


## Sharing Configurations

The Text and Preprocessing configurations should only be defined once per dataset and shared between your models to ensure each model makes the same assumptions about your data.
To achieve that, each model configuration can also be defined as a path to a configuration file. So, a configuration for a text-to-spec model that uses separately defined text and audio preprocessing configurations might look like this:

```yaml hl_lines="8 9"
model:
    decoder: ...
    ...
training:
    batch_size: 16
    ckpt_epochs: 1
    ...
path_to_preprocessing_config_file: "./config/default/{{ config_filename('preprocessing') }}"
path_to_text_config_file: "./config/default/{{ config_filename('text') }}"
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

### Symbols

Your symbol set is created by taking the union of all values defined. For example:

```yaml
symbols:
    dataset_0_characters: ['a', 'b', 'c']
    dataset_1_characters: ['b', 'c', 'd']
```

Will create a symbol set equal to `{'a', 'b', 'c', 'd'}` (i.e. the union of both key/values). This allows you to train models with data from different languages, for example.

!!! important
    You should always manually inspect your configuration here to make sure it makes sense with respect to your data. Is there a symbol that shouldn't be there? Is there a symbol that's defined as 'punctuation' but is used as non-punctuation in your language? Please inspect these and update the configuration accordingly.

::: everyvoice.config.text_config.Symbols
    handler: python
    options:
        show_root_heading: true
        heading_level: 6
