"""These are the Base model hyperparameters.

   TODO: Add intellisense (maybe add all keys as attributes)
   TODO: Give examples of creating alternate configurations
   TODO: Give examples of changing hyperparameters in CLI
"""

from copy import deepcopy
from datetime import datetime
from string import ascii_lowercase, ascii_uppercase

from smts.utils import (
    collapse_whitespace,
    generic_dict_loader,
    load_lj_metadata_hifigan,
    lower,
    nfc_normalize,
    original_hifigan_leaky_relu,
    rel_path_to_abs_path,
    update_config,
)

#########################
#                       #
#         MODEL         #
#    HYPERPARAMETERS    #
#                       #
#########################

BASE_MODEL_HPARAMS = {
    "aligner": {
        "lstm_dim": 512,
        "conv_dim": 512,
    },
    "transformer": {
        "encoder_layers": 4,
        "encoder_head": 2,
        "encoder_hidden": 256,
        "encoder_dim_feedforward": 1024,
        "encoder_conv_filter_size": 1024,
        "encoder_conv_kernel_sizes": [9, 1],
        "encoder_dropout": 0.2,
        "encoder_depthwise": True,
        "decoder_layers": 6,
        "decoder_head": 2,
        "decoder_hidden": 256,
        "decoder_dim_feedforward": 1024,
        "decoder_conv_filter_size": 1024,
        "decoder_conv_kernel_sizes": [9, 1],
        "decoder_dropout": 0.2,
        "decoder_depthwise": True,
        "num_phon_feats": 37,
        "use_phon_feats": False,
        "use_conformer": {
            "encoder": True,
            "decoder": True,
        }
        # etc...
    },
    "variance_adaptor": {
        "variances": ["pitch", "energy"],
        "variance_levels": ["phone", "phone"],  # frame or phone
        "variance_transforms": ["none", "none"],  # "cwt", "log", "none"
        "variance_losses": ["mse", "mse"],
        "variance_nlayers": [5, 5, 5, 5],
        "variance_loss_weights": [5e-2, 5e-2, 5e-2, 5e-2],
        "variance_kernel_size": [3, 3, 3, 3],
        "variance_dropout": [0.5, 0.5, 0.5, 0.5],
        "variance_filter_size": 256,
        "variance_nbins": 256,
        "variance_depthwise_conv": True,
        "filter_size": 256,
        "kernel_size": 3,
        "dropout": 0.5,
        "duration_nlayers": 2,
        "duration_stochastic": False,
        "duration_kernel_size": 3,
        "duration_dropout": 0.5,
        "duration_filter_size": 256,
        "duration_depthwise_conv": True,
        "soft_dtw_gamma": 0.1,
        "soft_dtw_chunk_size": 256
        # etc..
    },
    "feature_prediction": {
        "learn_alignment": True,  # True for Badlani et. al. 2021 Attention, False for Aligner-extracted durations
        "max_length": 1000,
    },
    "feature_prediction_loss": {
        "mel_loss": "mse",
        "mel_loss_weight": 1,
        "duration_loss": "mse",
        "duration_loss_weight": 5e-1,
    },
    "priors": {
        "prior_types": [
            "pitch",
            "energy",
            "duration",
        ],  # ["pitch", "energy", "duration"]
        "gmm": False,
        "gmm_max_components": 5,
        "gmm_min_samples_per_component": 20,
        "gmm_reg_covar": 1e-3,
        "gmm_logs": [0, 1, 2, 3],
        "every_layer": False,
    },
    "vocoder": {
        "resblock": "1",
        "upsample_rates": [
            8,
            8,
            2,
            2,
        ],  # 8, 8, 2, 2 preserves input sampling rate. 8, 8, 4, 2 doubles it for example.
        "upsample_kernel_sizes": [
            16,
            16,
            4,
            4,
        ],  # must not be less than upsample rate, and must be evenly divisible by upsample rate
        "upsample_initial_channel": 512,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "depthwise_separable_convolutions": {
            "generator": False,
        },
        "activation_function": original_hifigan_leaky_relu,  # for original implementation use utils.original_hifigan_leaky_relu,
        "istft_layer": False,  # Uses C8C8I model https://arxiv.org/pdf/2203.02395.pdf - must change upsample rates and upsample_kernel_sizes appropriately.
    },
    "use_postnet": True,
    "multilingual": True,
    "multispeaker": {
        "embedding_type": "id",  # "id", "dvector", None
        "every_layer": False,
        "dvector_gmm": False,
    },
}

#########################
#                       #
#       TRAINING        #
#    HYPERPARAMETERS    #
#                       #
#########################

BASE_TRAINING_HPARAMS = {
    "aligner": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "train_split": 0.99,
        "save_top_k_ckpts": 3,
        "binned_sampler": True,
        "seed": 1234,
        "max_epochs": 1000,
        "plot_steps": 1000,
        "ckpt_steps": None,
        "ckpt_epochs": 1,
        "finetune_checkpoint": "",
    },
    "training_strategy": "feat",  # "feat", "variance", "vocoder", "e2e"
    "train_data_workers": 0,
    "val_data_workers": 0,
    "logger": {  # Uses Tensorboard
        "name": "Base Experiment",
        "save_dir": rel_path_to_abs_path("./logs"),
        "sub_dir": str(int(datetime.today().timestamp())),
        "version": "alignment-sanity",
    },
    "feature_prediction": {
        "batch_size": 2,
        "train_split": 0.9,  # the rest is val
        "seed": 1234,
        # "filelist": rel_path_to_abs_path(
        #     "./preprocessed/YourDataSet/processed_filelist.psv"
        # ),
        "filelist": rel_path_to_abs_path("./preprocessed/LJ/processed_filelist.psv"),
        "filelist_loader": generic_dict_loader,
        "finetune_checkpoint": "",
        "optimizer": {
            "lr": 1e-4,
            "betas": (0.9, 0.98),
            "eps": 1e-8,
            "weight_decay": 0.01,
            "warmup_steps": 4000,
            # etc....
        },
        "freeze_layers": {
            "encoder": False,
            "decoder": False,
            "postnet": False,
            "variance": {"energy": False, "duration": False, "pitch": False},
        },
        "early_stopping": {
            "metric": "none",  # "none", "mae", or "js"
            "patience": 4,
        },
        "tf": {
            "ratio": 1.0,
            "linear_schedule": False,
            "linear_schedule_start": 0,
            "linear_schedule_end": 20,
            "linear_schedule_end_ratio": 0.0,
        },
        "max_epochs": 1000,
        "save_top_k_ckpts": 5,
        "ckpt_steps": None,
        "ckpt_epochs": 1,
        "use_weighted_sampler": False,
    },
    "vocoder": {
        "batch_size": 16,
        "train_split": 0.9,  # the rest is val
        "filelist": rel_path_to_abs_path(
            "./preprocessed/YourDataSet/processed_filelist.psv"
        ),
        "filelist_loader": generic_dict_loader,
        "finetune_checkpoint": rel_path_to_abs_path(
            "./logs/Base Experiment/sanity/checkpoints/last.ckpt"
        ),
        "resblock": "1",
        "learning_rate": 0.0002,
        "adam_b1": 0.8,
        "adam_b2": 0.99,
        "lr_decay": 0.999,
        "seed": 1234,
        "freeze_layers": {"mpd": False, "msd": False, "generator": False},
        "max_epochs": 1000,
        "save_top_k_ckpts": 5,
        "ckpt_steps": None,
        "ckpt_epochs": 1,
        "generator_warmup": 0,
        "gan_type": "original",  # original, wgan, wgan-gp
        "gan_optimizer": "adam",  # adam, rmsprop
        "wgan_clip_value": 0.01,
        "use_weighted_sampler": False,
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
    "source_data": [
        {
            "label": "LJ",
            "data_dir": "/home/aip000/tts/corpora/Speech/LJ.Speech.Dataset/LJSpeech-1.1/wavs",
            "filelist_loader": load_lj_metadata_hifigan,
            "filelist": rel_path_to_abs_path("./filelists/lj_full.psv"),
            "sox_effects": SOX_EFFECTS,
        }
    ],
    "save_dir": rel_path_to_abs_path("./preprocessed/LJ"),
    # "source_data": [
    #     {
    #         "label": "LJ_TEST",
    #         "data_dir": rel_path_to_abs_path("./tests/data/lj/wavs"),
    #         "textgrid_dir": rel_path_to_abs_path("./tests/data/lj/textgrids"),
    #         "filelist_loader": load_lj_metadata_hifigan,
    #         "filelist": rel_path_to_abs_path("./filelists/lj_full.psv"),
    #         "sox_effects": SOX_EFFECTS,
    #     }
    # ],
    # "save_dir": rel_path_to_abs_path("./preprocessed/YourDataSet"),
    "pitch_phone_averaging": True,
    "energy_phone_averaging": True,
    "pitch_type": "pyworld",  # pyworld | kaldi (torchaudio) | cwt (continuous wavelet transform)
    "value_separator": "--",  # used to separate basename from speaker, language, type etc in preprocessed filename
    "audio": {
        "min_audio_length": 0.25,  # seconds
        "max_audio_length": 11,  # seconds
        "max_wav_value": 32768.0,
        "norm_db": -3.0,
        "sil_threshold": 1.0,
        "sil_duration": 0.1,
        "input_sampling_rate": 22050,  # Sampling rate to ensure audio input to vocoder (output spec from feature prediction) is sampled at
        "output_sampling_rate": 22050,  # Sampling rate to ensure audio output of vocoder is sampled at
        "target_bit_depth": 16,
        "alignment_sampling_rate": 22050,  # Sampling rate from TextGrids. These two sampling rates *should* be the same, but they are separated in case it's not practical for your data
        "alignment_bit_depth": 16,
        "fft_window_frames": 1024,  # set this to the input sampling rate
        "fft_hop_frames": 256,  # set this to the input sampling rate
        "f_min": 0,
        "f_max": 8000,
        "n_fft": 1024,  # set this to the input sampling rate
        "n_mels": 80,
        "spec_type": "mel-librosa",  # mel (real) | linear (real) | raw (complex) see https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html#overview-of-audio-features
        "vocoder_segment_size": 8192,  # this is the size of the segments taken for training HiFI-GAN. set proportional to output sampling rate; 8192 is for output of 22050Hz. This should be a multiple of the upsample hop size which itself is equal to the product of the upsample rates.
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
# a list of strings.
# Note there is a limit with MLFlow for 250 characters per category: https://github.com/mlflow/mlflow/issues/6183
SYMBOLS = {
    "silence": ["<SIL>"],
    "pad": "_",
    "punctuation": "-';:,.!?¡¿—…\"«»“” ",
    "lowercase_letters": list(ascii_lowercase),
    "uppercase_letters": list(ascii_uppercase),
}

# Cleaners are defined in the configuration as opposed to editing a cleaners.py file somewhere else.
# Functions are applied to text in sequence.
# Unfortunately, you can't use lambda functions, because PyTorch doesn't support it. See https://github.com/pytorch/pytorch/issues/13300
CLEANERS = [lower, collapse_whitespace, nfc_normalize]


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
        config = update_config(base, kwargs)
        # then iter through and update from each arg
        for arg in args:
            if isinstance(arg, dict):
                config = update_config(config, arg)
            else:
                raise ValueError(
                    "Hm, you tried to update the config with an argument that was not a dictionary, please re-formulate"
                )
        super().__init__(*args, **config)
