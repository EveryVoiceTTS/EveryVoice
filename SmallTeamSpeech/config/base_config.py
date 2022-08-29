"""These are the Base model hyperparameters.

   TODO: Add intellisense (maybe add all keys as attributes)
   TODO: Give examples of creating alternate configurations
   TODO: Give examples of changing hyperparameters in CLI
"""


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
BASE_PREPROCESSING_HPARAMS = {}


class BaseConfig(dict):
    def __init__(self, *args, **kwargs):
        # Allow BaseConfig to be initialized with values that override the defaults
        kwargs = {
            **{
                "model": BASE_MODEL_HPARAMS,
                "training": BASE_TRAINING_HPARAMS,
                "preprocessing": BASE_PREPROCESSING_HPARAMS,
            },
            **kwargs,
        }
        super().__init__(*args, **kwargs)
