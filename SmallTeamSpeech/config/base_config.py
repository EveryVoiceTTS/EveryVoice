"""These are the Base model hyperparameters.

   TODO: Add intellisense (maybe add all keys as attributes)
   TODO: Give examples of creating alternate configurations
   TODO: Give examples of changing hyperparameters in CLI
"""

from collections.abc import Mapping
from copy import deepcopy
from string import ascii_letters
from unicodedata import normalize

from utils import collapse_whitespace

#########################
#                       #
#         MODEL         #
#    HYPERPARAMETERS    #
#                       #
#########################

BASE_MODEL_HPARAMS = {
    "encoder": {
        "encoder_layer": 4
        # etc...
    },
    "decoder": {},
    "variance_predictor": {
        # etc..
    },
    "vocoder": {
        # etc...
    },
    "use_postnet": True,
    "max_seq_len": 1000,
    "multilingual": True,
    "multispeaker": {
        # etc...
    },
}

#########################
#                       #
#       TRAINING        #
#    HYPERPARAMETERS    #
#                       #
#########################

BASE_TRAINING_HPARAMS = {
    "output_path": "./output",
    "steps": {
        "total": 300000,
        "log": 100,
        "synth": 1000,
        "val": 1000,
        "save": 100000,
    },
    "optimizer": {
        "batch_size": 16,
        # etc....
    },
}

#########################
#                       #
#     PREPROCESSING     #
#    HYPERPARAMETERS    #
#                       #
#########################

# These effects are applied to the audio during preprocessing
SOX_EFFECTS = [
    ["channels", "1"],  # convert to mono
    ["rate", "16000"],  # resample
    ["norm", "-3.0"],  # normalize to -3 dB
    [
        "silence",
        "1",  # Above periods silence; ie. allow 1 second of silence at beginning
        "0.1",  # Above periods silence duration
        "1.0%",  # Above periods silence threshold
        "-1",  # See https://linux.die.net/man/1/sox#:~:text=To%20remove%20silence,of%20the%20audio.
        "0.1",  # Below periods silence duration
        "1.0%",  # Below periods silence threshold
    ],  # remove silence throughout the file
]

BASE_PREPROCESSING_HPARAMS = {
    "dataset": "YourDataSet",
    "f0_phone_averaging": True,
    "energy_phone_averaging": True,
    "f0_type": "torch",  # pyworld | kaldi (torchaudio) | cwt (continuous wavelet transform)
    "audio": {
        "norm_db": -3.0,
        "sil_threshold": 1.0,
        "sil_duration": 0.1,
        "target_sampling_rate": 22050,  # Sampling rate to ensure audio is sampled using for inputs
        "alignment_sampling_rate": 22050,  # Sampling rate from TextGrids. These two sampling rates *should* be the same, but they are separated in case it's not practical for your data
        "fft_window_frames": 1024,
        "fft_hop_frames": 256,
        "f_min": 0,
        "f_max": 8000,
        "n_fft": 1024,
        "n_mels": 80,
        "spec_type": "mel",  # mel (real) | linear (real) | raw (complex) see https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html#overview-of-audio-features
        "sox_effects": SOX_EFFECTS,
    },
}

#########################
#                       #
#   TEXT CONFIGURATION  #
#                       #
#########################


# These values can be overwritten, and the keys are not specifically named,
# so you can add other keys to this dictionary (i.e. "ipa_characters" or other such key names)
# the value of each key here will be turned into a list, so make sure your symbol definitions
# are iterable. And, make sure that if you have digraphs/multigraphs, that they are defined as
# a list of strings
SYMBOLS = {
    "pad": "_",
    "punctuation": ';:,.!?¡¿—…"«»“” ',
    "letters": list(ascii_letters),
}

# Cleaners are defined in the configuration as opposed to editing a cleaners.py file somewhere else.
# Functions are applied to text in sequence
CLEANERS = [
    lambda text: text.lower(),
    collapse_whitespace,
    lambda text: normalize("NFC", text),
]


class BaseConfig(dict):
    def __init__(self, *args, **kwargs):
        # Allow BaseConfig to be initialized with values that override the defaults
        # pass by value to not change base hparams
        base = {
            "model": deepcopy(BASE_MODEL_HPARAMS),
            "training": deepcopy(BASE_TRAINING_HPARAMS),
            "preprocessing": deepcopy(BASE_PREPROCESSING_HPARAMS),
            "text": {"symbols": deepcopy(SYMBOLS), "cleaners": CLEANERS},
        }
        # update from kwargs
        config = update(base, kwargs)
        # then iter through and update from each arg
        for arg in args:
            if isinstance(arg, dict):
                config = update(config, arg)
            else:
                raise ValueError(
                    "Hm, you tried to update the config with an argument that was not a dictionary, please re-formulate"
                )
        super().__init__(*args, **config)


def update(orig_dict, new_dict):
    """See https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"""
    for key, val in new_dict.items():
        if isinstance(val, Mapping):
            tmp = update(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = orig_dict.get(key, []) + val
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict
