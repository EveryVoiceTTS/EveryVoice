import json
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import KernelDensity
from torch import nn
from tqdm import tqdm

from smts.config.base_config import BaseConfig
from smts.model.feature_prediction.layers import (
    ConformerEncoderLayer,
    PositionalEncoding,
    PriorEmbedding,
    SpeakerEmbedding,
    VarianceAdaptor,
)
from smts.model.feature_prediction.log_gmm import LogGMM
from smts.model.feature_prediction.loss import FastSpeech2Loss
from smts.model.feature_prediction.noam import NoamLR
from smts.model.vocoder.hifigan import HiFiGAN
from smts.text import TextProcessor
from smts.text.lookups import LookupTables


class FastSpeech2(pl.LightningModule):
    def __init__(
        self,
        config: BaseConfig,
    ):
        super().__init__()
        self.config = config
        self.batch_size = config["training"]["feature_prediction"]["batch_size"]
        # TODO: Why specific early stopping directory?
        # if variance_early_stopping != "none":
        #     letters = string.ascii_lowercase
        #     random_dir = "".join(random.choice(letters) for i in range(10))
        #     self.variance_encoder_dir = (
        #         Path(variance_early_stopping_directory) / random_dir
        #     )
        #     self.variance_encoder_dir.mkdir(parents=True, exist_ok=True)
        self.lookup = LookupTables(config)
        self.speaker2id = self.lookup.speaker2id
        self.lang2id = self.lookup.lang2id
        self.text_processor = TextProcessor(config)
        # hparams
        self.save_hyperparameters(
            ignore=[
                "train_ds",
                "valid_ds",
                "train_ds_kwargs",
                "valid_ds_kwargs",
                "valid_nexamples",
                "valid_example_directory",
                "batch_size",
                "variance_early_stopping_directory",
                "num_workers",
            ]
        )

        self.eval_log_data = None
        with open(
            self.config["preprocessing"]["save_dir"] / "stats.json", encoding="UTF-8"
        ) as f:
            self.stats = json.load(f)
        # # data # TODO: refactor to use config
        # TODO: Why this? Just load datasets...
        # if train_ds is not None:
        #     if train_ds_kwargs is None:
        #         train_ds_kwargs = {}
        #     train_ds_kwargs["speaker_type"] = speaker_type
        #     train_ds_kwargs["min_length"] = min_length
        #     train_ds_kwargs["max_length"] = max_length
        #     train_ds_kwargs["augment_duration"] = augment_duration
        #     train_ds_kwargs["variances"] = variances
        #     train_ds_kwargs["variance_levels"] = variance_levels
        #     train_ds_kwargs["variance_transforms"] = variance_transforms
        #     train_ds_kwargs["priors"] = priors
        #     train_ds_kwargs["n_mels"] = n_mels
        #     train_ds_kwargs["sampling_rate"] = sampling_rate
        #     train_ds_kwargs["n_fft"] = n_fft
        #     train_ds_kwargs["win_length"] = win_length
        #     train_ds_kwargs["hop_length"] = hop_length
        #     if cache_path is not None:
        #         if isinstance(train_ds, list):
        #             hashes = [x.hash for x in train_ds]
        #         else:
        #             hashes = [train_ds.hash]
        #         kwargs = copy(train_ds_kwargs)
        #         kwargs.update({"hashes": hashes})
        #         ds_hash = hashlib.md5(
        #             json.dumps(kwargs, sort_keys=True).encode("utf-8")
        #         ).hexdigest()
        #         cache_path_full = Path(cache_path) / f"train-full-{ds_hash}.pt"
        #         if cache_path_full.exists():
        #             with cache_path_full.open("rb") as f:
        #                 self.train_ds = pickle.load(f)
        #         else:
        #             self.train_ds = TTSDataset(train_ds, **train_ds_kwargs)
        #             self.train_ds.hash = ds_hash
        #             with open(cache_path_full, "wb") as f:
        #                 pickle.dump(self.train_ds, f)
        #     else:
        #         self.train_ds = TTSDataset(train_ds, **train_ds_kwargs)
        # if valid_ds is not None:
        #     if valid_ds_kwargs is None:
        #         valid_ds_kwargs = {}
        #     if cache_path is not None:
        #         kwargs = copy(valid_ds_kwargs)
        #         kwargs.update({"hashes": [self.train_ds.hash, valid_ds.hash]})
        #         ds_hash = hashlib.md5(
        #             json.dumps(kwargs, sort_keys=True).encode("utf-8")
        #         ).hexdigest()
        #         cache_path_full = Path(cache_path) / f"valid-full-{ds_hash}.pt"
        #         if cache_path_full.exists():
        #             with cache_path_full.open("rb") as f:
        #                 self.valid_ds = pickle.load(f)
        #         else:
        #             self.valid_ds = self.train_ds.create_validation_dataset(
        #                 valid_ds, **valid_ds_kwargs
        #             )
        #             self.valid_ds.hash = ds_hash
        #             with open(cache_path_full, "wb") as f:
        #                 pickle.dump(self.valid_ds, f)
        #     else:
        #         self.valid_ds = self.train_ds.create_validation_dataset(
        #             valid_ds, **valid_ds_kwargs
        #         )
        self.vocoder = HiFiGAN(
            config=config
        )  # TODO: make this general for other vocoders potentially
        if self.config["training"]["training_strategy"] != "e2e":
            self.vocoder.freeze()

        # needed for inference without a dataset
        # if train_ds is not None:
        #     self.stats = self.train_ds.stats
        #     self.phone2id = self.train_ds.phone2id
        #     if "dvector" in self.train_ds.speaker_type:
        #         self.speaker2dvector = self.train_ds.speaker2dvector
        #     if self.train_ds.speaker_type == "id":
        #         self.speaker2id = self.train_ds.speaker2id

        if self.config["model"]["transformer"]["use_phon_feats"]:
            self.text_input_layer = nn.Linear(
                self.config["model"]["transformer"]["num_phon_feats"],
                self.config["model"]["transformer"]["encoder_hidden"],
                bias=False,
            )
        else:
            self.text_input_layer = nn.Embedding(
                len(self.text_processor.symbols),
                self.config["model"]["transformer"]["encoder_hidden"],
                padding_idx=0,
            )

        # encoder

        self.encoder = nn.TransformerEncoder(
            ConformerEncoderLayer(
                self.config["model"]["transformer"]["encoder_hidden"],
                self.config["model"]["transformer"]["encoder_head"],
                conv_in=self.config["model"]["transformer"]["encoder_hidden"],
                conv_filter_size=self.config["model"]["transformer"][
                    "encoder_conv_filter_size"
                ],
                conv_kernel=(
                    self.config["model"]["transformer"]["encoder_conv_kernel_sizes"][0],
                    1,
                ),
                batch_first=True,
                dropout=self.config["model"]["transformer"]["encoder_dropout"],
                conv_depthwise=self.config["model"]["transformer"]["encoder_depthwise"],
            ),
            num_layers=self.config["model"]["transformer"]["encoder_layers"],
        )

        if self.config["model"]["transformer"]["use_conformer"]["encoder"]:
            if (
                self.config["model"]["transformer"]["use_conformer"]["encoder"]
                is not None
            ):
                print("encoder_dim_feedforward is ignored for conformer")
            for i in range(self.config["model"]["transformer"]["encoder_layers"]):
                self.encoder.layers[i] = ConformerEncoderLayer(
                    self.config["model"]["transformer"]["encoder_hidden"],
                    self.config["model"]["transformer"]["encoder_head"],
                    conv_in=self.config["model"]["transformer"]["encoder_hidden"],
                    conv_filter_size=self.config["model"]["transformer"][
                        "encoder_conv_filter_size"
                    ],
                    conv_kernel=(
                        self.config["model"]["transformer"][
                            "encoder_conv_kernel_sizes"
                        ][0],
                        1,
                    ),
                    batch_first=True,
                    dropout=self.config["model"]["transformer"]["encoder_dropout"],
                    conv_depthwise=self.config["model"]["transformer"][
                        "encoder_depthwise"
                    ],
                )
        else:
            for i in range(self.config["model"]["transformer"]["encoder_layers"]):
                self.encoder.layers[i] = nn.TransformerEncoderLayer(
                    self.config["model"]["transformer"]["encoder_hidden"],
                    self.config["model"]["transformer"]["encoder_head"],
                    dim_feedforward=self.config["model"]["transformer"][
                        "encoder_dim_feedforward"
                    ],
                    batch_first=True,
                    dropout=self.config["model"]["transformer"]["encoder_dropout"],
                )
        self.positional_encoding = PositionalEncoding(
            self.config["model"]["transformer"]["encoder_hidden"],
            dropout=self.config["model"]["transformer"]["encoder_dropout"],
        )

        # variances
        if hasattr(self, "stats"):
            self.variance_adaptor = VarianceAdaptor(
                self.stats,
                self.config["model"]["variance_adaptor"]["variances"],
                self.config["model"]["variance_adaptor"]["variance_levels"],
                self.config["model"]["variance_adaptor"]["variance_transforms"],
                self.config["model"]["variance_adaptor"]["variance_nlayers"],
                self.config["model"]["variance_adaptor"]["variance_kernel_size"],
                self.config["model"]["variance_adaptor"]["variance_dropout"],
                self.config["model"]["variance_adaptor"]["variance_filter_size"],
                self.config["model"]["variance_adaptor"]["variance_nbins"],
                self.config["model"]["variance_adaptor"]["variance_depthwise_conv"],
                self.config["model"]["variance_adaptor"]["duration_nlayers"],
                self.config["model"]["variance_adaptor"]["duration_stochastic"],
                self.config["model"]["variance_adaptor"]["duration_kernel_size"],
                self.config["model"]["variance_adaptor"]["duration_dropout"],
                self.config["model"]["variance_adaptor"]["duration_filter_size"],
                self.config["model"]["variance_adaptor"]["duration_depthwise_conv"],
                self.config["model"]["transformer"]["encoder_hidden"],
                self.config["model"]["feature_prediction"]["max_length"]
                * self.config["preprocessing"]["audio"]["input_sampling_rate"]
                / self.config["preprocessing"]["audio"]["fft_hop_frames"],
            ).to(self.device)

        # decoder
        self.decoder = nn.TransformerEncoder(
            ConformerEncoderLayer(
                self.config["model"]["transformer"]["decoder_hidden"],
                self.config["model"]["transformer"]["decoder_head"],
                conv_in=self.config["model"]["transformer"]["decoder_hidden"],
                conv_filter_size=self.config["model"]["transformer"][
                    "decoder_conv_filter_size"
                ],
                conv_kernel=(
                    self.config["model"]["transformer"]["decoder_conv_kernel_sizes"][0],
                    1,
                ),
                batch_first=True,
                dropout=self.config["model"]["transformer"]["decoder_dropout"],
                conv_depthwise=self.config["model"]["transformer"]["decoder_depthwise"],
            ),
            num_layers=self.config["model"]["transformer"]["decoder_layers"],
        )
        if self.config["model"]["transformer"]["use_conformer"]["decoder"]:
            if (
                self.config["model"]["transformer"]["decoder_dim_feedforward"]
                is not None
            ):
                print("decoder_dim_feedforward is ignored for conformer")
            for i in range(self.config["model"]["transformer"]["decoder_layers"]):
                self.decoder.layers[i] = ConformerEncoderLayer(
                    self.config["model"]["transformer"]["decoder_hidden"],
                    self.config["model"]["transformer"]["decoder_head"],
                    conv_in=self.config["model"]["transformer"]["decoder_hidden"],
                    conv_filter_size=self.config["model"]["transformer"][
                        "decoder_conv_filter_size"
                    ],
                    conv_kernel=(
                        self.config["model"]["transformer"][
                            "decoder_conv_kernel_sizes"
                        ][0],
                        1,
                    ),
                    batch_first=True,
                    dropout=self.config["model"]["transformer"]["decoder_dropout"],
                    conv_depthwise=self.config["model"]["transformer"][
                        "decoder_depthwise"
                    ],
                )
        else:
            for i in range(self.config["model"]["transformer"]["decoder_layers"]):
                self.decoder.layers[i] = nn.TransformerEncoderLayer(
                    self.config["model"]["transformer"]["decoder_hidden"],
                    self.config["model"]["transformer"]["decoder_head"],
                    dim_feedforward=self.config["model"]["transformer"][
                        "decoder_dim_feedforward"
                    ],
                    batch_first=True,
                    dropout=self.config["model"]["transformer"]["decoder_dropout"],
                )

        self.linear = nn.Linear(
            self.config["model"]["transformer"]["decoder_hidden"],
            self.config["preprocessing"]["audio"]["n_mels"],
        )

        # priors
        if hasattr(self, "stats"):
            self.prior_embeddings = {}
            for prior in self.config["model"]["priors"]["prior_types"]:
                self.prior_embeddings[prior] = PriorEmbedding(
                    self.config["model"]["transformer"]["encoder_hidden"],
                    self.config["model"]["variance_adaptor"]["variance_nbins"],
                    self.stats[f"{prior}_prior"],
                ).to(self.device)
            self.prior_embeddings = nn.ModuleDict(self.prior_embeddings)

        # speaker
        if self.config["model"]["multispeaker"]["embedding_type"] == "dvector":
            self.speaker_embedding = SpeakerEmbedding(
                self.config["model"]["transformer"]["encoder_hidden"],
                self.config["model"]["multispeaker"]["embedding_type"],
            )
        elif self.config["model"]["multispeaker"]["embedding_type"] == "id":
            if hasattr(self, "speaker2id"):
                self.speaker_embedding = SpeakerEmbedding(
                    self.config["model"]["transformer"]["decoder_hidden"],
                    self.config["model"]["multispeaker"]["embedding_type"],
                    len(self.speaker2id),  # TODO: create speaker2id embedding
                )
        if self.config["model"]["multilingual"]:
            self.lang_embedding = nn.Embedding(
                len(self.lang2id), self.config["model"]["transformer"]["decoder_hidden"]
            )
        if (
            hasattr(self, "train_ds")
            and self.train_ds is not None
            and hasattr(self.train_ds, "speaker2priors")  # TODO: create speaker2priors
        ):
            self.speaker2priors = self.train_ds.speaker_priors

        # loss
        loss_weights = {
            "mel": self.config["model"]["feature_prediction_loss"]["mel_loss_weight"],
            "duration": self.config["model"]["feature_prediction_loss"][
                "duration_loss_weight"
            ],
        }
        for i, var in enumerate(self.config["model"]["variance_adaptor"]["variances"]):
            loss_weights[var] = self.config["model"]["variance_adaptor"][
                "variance_loss_weights"
            ][i]
        self.loss = FastSpeech2Loss(
            self.config["model"]["variance_adaptor"]["variances"],
            self.config["model"]["variance_adaptor"]["variance_levels"],
            self.config["model"]["variance_adaptor"]["variance_transforms"],
            self.config["model"]["variance_adaptor"]["variance_losses"],
            self.config["model"]["feature_prediction_loss"]["mel_loss"],
            self.config["model"]["feature_prediction_loss"]["duration_loss"],
            self.config["model"]["variance_adaptor"]["duration_stochastic"],
            self.config["model"]["feature_prediction"]["max_length"]
            * self.config["preprocessing"]["audio"]["input_sampling_rate"]
            / self.config["preprocessing"]["audio"]["fft_hop_frames"],
            loss_weights,
            self.config["model"]["variance_adaptor"]["soft_dtw_gamma"],
            self.config["model"]["variance_adaptor"]["soft_dtw_chunk_size"],
        )

        if (
            len(self.config["model"]["priors"]["prior_types"]) > 0
            and self.config["model"]["priors"]["gmm"]
            and hasattr(self, "speaker2priors")
        ):
            self._fit_speaker_prior_gmms()

        if (
            self.config["model"]["multispeaker"]["dvector_gmm"]
            and self.train_ds is not None
            and hasattr(self.train_ds, "speaker2dvector")
        ):
            self._fit_speaker_dvector_gmms()

        elif self.config["model"]["multispeaker"]["dvector_gmm"]:
            print(
                "dvector_gmm is ignored because train_ds is not provided or speaker2dvector is not provided"
            )

    def _fit_speaker_dvector_gmms(self):
        self.dvector_gmms = {}
        for speaker, dvector in self.train_ds.get_speaker_dvectors():
            self.dvector_gmms[speaker] = LogGMM(n_components=10, random_state=0)
            self.dvector_gmms[speaker].fit(dvector)

    def _fit_speaker_prior_gmms(self):
        self.speaker_gmms = {}
        reg_covar = self.config["model"]["priors"]["gmm_reg_covar"]
        for speaker in tqdm(
            self.speaker2priors.keys(), desc="fitting speaker prior gmms"
        ):
            priors = self.speaker2priors[speaker]
            X = np.stack(
                [
                    priors[prior]
                    for prior in self.config["model"]["priors"]["prior_types"]
                ],
                axis=1,
            )
            gmm_kwargs = {
                "n_components": 1,
                "random_state": 0,
                "reg_covar": reg_covar,
                "logs": self.config["model"]["priors"]["gmm_logs"],
            }
            best_bic = np.infty
            while True:
                gmm = LogGMM(**gmm_kwargs)
                gmm.fit(X)
                bic = gmm.bic(X)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
                else:
                    break
                n_components = gmm_kwargs["n_components"]
                if (
                    n_components == len(X)
                    or n_components
                    >= self.config["model"]["priors"]["gmm_max_components"]
                    or self.config["model"]["priors"]["gmm_min_samples_per_component"]
                    * n_components
                    > len(X)
                ):
                    break
                gmm_kwargs["n_components"] += 1
            self.speaker_gmms[speaker] = best_gmm

    def on_load_checkpoint(self, checkpoint):
        self.stats = checkpoint["stats"]
        if not hasattr(self, "variance_adaptor"):
            self.variance_adaptor = VarianceAdaptor(
                self.stats,  # TODO: check stats
                self.config["model"]["variance_adaptor"]["variances"],
                self.config["model"]["variance_adaptor"]["variance_levels"],
                self.config["model"]["variance_adaptor"]["variance_transforms"],
                self.config["model"]["variance_adaptor"]["variance_nlayers"],
                self.config["model"]["variance_adaptor"]["variance_kernel_size"],
                self.config["model"]["variance_adaptor"]["variance_dropout"],
                self.config["model"]["variance_adaptor"]["variance_filter_size"],
                self.config["model"]["variance_adaptor"]["variance_nbins"],
                self.config["model"]["variance_adaptor"]["variance_depthwise_conv"],
                self.config["model"]["variance_adaptor"]["duration_nlayers"],
                self.config["model"]["variance_adaptor"]["duration_stochastic"],
                self.config["model"]["variance_adaptor"]["duration_kernel_size"],
                self.config["model"]["variance_adaptor"]["duration_dropout"],
                self.config["model"]["variance_adaptor"]["duration_filter_size"],
                self.config["model"]["variance_adaptor"]["duration_depthwise_conv"],
                self.config["model"]["transformer"]["encoder_hidden"],
                self.config["model"]["feature_prediction"]["max_length"]
                * self.config["preprocessing"]["audio"]["input_sampling_rate"]
                / self.config["preprocessing"]["audio"]["fft_hop_frames"],
            ).to(self.device)
        if not hasattr(self, "prior_embeddings"):
            self.prior_embeddings = {}
            for prior in self.config["model"]["priors"]["prior_types"]:
                self.prior_embeddings[prior] = PriorEmbedding(
                    self.config["model"]["transformer"]["encoder_hidden"],
                    self.config["model"]["variance_adaptor"]["variance_nbins"],
                    self.stats[f"{prior}_prior"],
                ).to(self.device)
            self.prior_embeddings = nn.ModuleDict(self.prior_embeddings)
        self.phone2id = checkpoint["phone2id"]
        # if not hasattr(self, "phone_embedding"):
        #     self.phone_embedding = nn.Embedding(
        #         len(self.phone2id),
        #         self.config["model"]["transformer"]["encoder_hidden"],
        #         padding_idx=0,
        #     )
        if "speaker2id" in checkpoint:
            self.speaker2id = checkpoint["speaker2id"]
            if not hasattr(self, "speaker_embedding"):
                self.speaker_embedding = SpeakerEmbedding(
                    self.config["model"]["transformer"]["encoder_hidden"],
                    self.config["model"]["multispeaker"]["embedding_type"],
                    len(self.speaker2id),
                )
        if "speaker2dvector" in checkpoint:
            self.speaker2dvector = checkpoint["speaker2dvector"]
        if "speaker2priors" in checkpoint:
            self.speaker2priors = checkpoint["speaker2priors"]
        if "speaker_gmms" in checkpoint:
            self.speaker_gmms = checkpoint["speaker_gmms"]
        if "dvector_gmms" in checkpoint:
            self.dvector_gmms = checkpoint["dvector_gmms"]

        # drop shape mismatched layers
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            # pylint: disable=unsupported-membership-test,unsubscriptable-object
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            # pylint: enable=unsupported-membership-test,unsubscriptable-object
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["stats"] = self.stats
        checkpoint["phone2id"] = self.phone2id
        if hasattr(self, "speaker2id"):
            checkpoint["speaker2id"] = self.speaker2id
        if hasattr(self, "speaker2dvector"):
            checkpoint["speaker2dvector"] = self.speaker2dvector
        if hasattr(self, "speaker2priors"):
            checkpoint["speaker2priors"] = self.speaker2priors
        if hasattr(self, "speaker_gmms"):
            checkpoint["speaker_gmms"] = self.speaker_gmms
        if hasattr(self, "dvector_gmms"):
            checkpoint["dvector_gmms"] = self.dvector_gmms

    def forward(self, batch, optimizer_idx=0, inference=False):
        text = batch["text"].to(self.device)
        speakers = batch["speaker_id"].to(self.device)

        src_mask = text.eq(0)  # TODO: map to silence/pad?

        output = self.text_input_layer(text)

        output = self.positional_encoding(output)

        if not self.config["model"]["multispeaker"]["every_layer"]:
            output = output + self.speaker_embedding(
                speakers, output.shape[1], output.shape[-1]
            )
        else:
            every_encoder_layer = self.speaker_embedding(
                speakers, output.shape[1], output.shape[-1]
            )
        # CHECKED UP TO HERE
        if self.config["model"]["priors"]["every_layer"]:
            # TODO: UNCHECKED
            for prior in self.config["model"]["priors"]["prior_types"]:
                every_encoder_layer = every_encoder_layer + self.prior_embeddings[
                    prior
                ](
                    torch.tensor(batch[f"priors_{prior}"]).to(self.device),
                    output.shape[1],
                )

        if (
            self.config["model"]["priors"]["every_layer"]
            or self.config["model"]["multispeaker"]["every_layer"]
        ):
            # TODO: UNCHECKED
            output = self.encoder(
                output,
                src_key_padding_mask=src_mask,
                additional_src=every_encoder_layer,
            )
        else:
            # TODO: CHECKED
            output = self.encoder(output, src_key_padding_mask=src_mask)

        if not self.config["model"]["priors"]["every_layer"]:
            for prior in self.config["model"]["priors"]["prior_types"]:
                # TODO: Broken
                output = output + self.prior_embeddings[prior](
                    torch.tensor(batch[f"priors_{prior}"]).to(self.device),
                    output.shape[1],
                )

        tf_ratio = self.config["training"]["feature_prediction"]["tf"]["ratio"]

        if self.config["training"]["feature_prediction"]["tf"]["linear_schedule"] and (
            self.current_epoch
            > self.config["training"]["feature_prediction"]["tf"][
                "linear_schedule_start"
            ]
        ):
            tf_ratio = tf_ratio - (
                (
                    tf_ratio
                    - self.config["training"]["feature_prediction"]["tf"][
                        "linear_schedule_end_ratio"
                    ]
                )
                * (
                    self.current_epoch
                    - self.config["training"]["feature_prediction"]["tf"][
                        "linear_schedule_start"
                    ]
                )
                / (
                    self.config["training"]["feature_prediction"]["tf"][
                        "linear_schedule_end"
                    ]
                    - self.config["training"]["feature_prediction"]["tf"][
                        "linear_schedule_start"
                    ]
                )
            )
        self.log("tf_ratio", tf_ratio)

        # pylint: disable=not-callable
        # TODO: Broken
        variance_output = self.variance_adaptor(
            output,
            src_mask,
            batch,
            inference=inference,
            tf_ratio=tf_ratio,
        )
        # pylint: enable=not-callable

        if optimizer_idx == 0:
            # TODO: Broken
            output = variance_output["x"]
            # TODO: CHECKED
            output = self.positional_encoding(output)

            if not self.config["model"]["multispeaker"]["every_layer"]:
                output = output + self.speaker_embedding(
                    speakers, output.shape[1], output.shape[-1]
                )
            else:
                every_decoder_layer = self.speaker_embedding(
                    speakers, output.shape[1], output.shape[-1]
                )

            if self.config["model"]["multispeaker"]["every_layer"]:
                output = self.decoder(
                    output,
                    src_key_padding_mask=variance_output["tgt_mask"],
                    additional_src=every_decoder_layer,
                )
            else:
                output = self.decoder(
                    output, src_key_padding_mask=variance_output["tgt_mask"]
                )

            output = self.linear(output)

            result = {
                "mel": output,
                "duration_prediction": variance_output["duration_prediction"],
                "duration_rounded": variance_output["duration_rounded"],
                "src_mask": src_mask,
                "tgt_mask": variance_output["tgt_mask"],
            }

            for var in self.config["model"]["variance_adaptor"]["variances"]:
                result[f"variances_{var}"] = variance_output[f"variances_{var}"]
                if f"variances_{var}_disc_true" in variance_output:
                    result[f"variances_{var}_disc_true"] = variance_output[
                        f"variances_{var}_disc_true"
                    ]
                    result[f"variances_{var}_disc_fake"] = variance_output[
                        f"variances_{var}_disc_fake"
                    ]

        elif optimizer_idx == 1:
            result = {
                f"variances_{var}_gen_true": variance_output[
                    f"variances_{var}_gen_true"
                ]
                for var in self.config["model"]["variance_adaptor"]["variances"]
            }

        return result

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        result = self(batch, optimizer_idx)
        losses = self.loss(
            result, batch
        )  # frozen_components=self.variance_adaptor.frozen_components)
        log_dict = {f"train/{k}_loss": v.item() for k, v in losses.items()}
        self.log_dict(
            log_dict,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        return losses["total"]

    def validation_step(self, batch, batch_idx):

        result = self(batch)
        losses = self.loss(result, batch)
        log_dict = {f"eval/{k}_loss": v.item() for k, v in losses.items()}
        self.log_dict(
            log_dict,
            batch_size=self.batch_size,
            sync_dist=True,
        )

        if batch_idx == 0 and self.trainer.is_global_zero:
            self.eval_log_data = []
            self.results_dict = {
                "duration": {"pred": [], "true": []},
                "mel": {"pred": [], "true": []},
            }
            for var in self.config["model"]["variance_adaptor"]["variances"]:
                self.results_dict[var] = {"pred": [], "true": []}

        inference_result = self(batch, inference=True)

        if (
            self.eval_log_data is not None
            and len(self.eval_log_data) < self.valid_nexamples
            and self.trainer.is_global_zero
        ):
            left_to_add = self.valid_nexamples - len(self.eval_log_data)
            self._add_to_results_dict(inference_result, batch, result, left_to_add)
            self._log_table_to_wandb(inference_result, batch, result)

    def _log_table_to_wandb(self, inference_result, batch, result):
        for i in range(len(batch["mel"])):
            if len(self.eval_log_data) >= self.valid_nexamples:
                break
            pred_mel = inference_result["mel"][i][
                ~inference_result["tgt_mask"][i]
            ].cpu()
            true_mel = batch["mel"][i][~result["tgt_mask"][i]].cpu()
            pred_dict = {
                "mel": pred_mel,
                "duration": inference_result["duration_rounded"][i][
                    ~inference_result["src_mask"][i]
                ].cpu(),
                "text": batch["text"][i],
                "raw_text": batch["raw_text"][i],
                "variances": {},
                "priors": {},
            }
            for j, var in enumerate(
                self.config["model"]["variance_adaptor"]["variances"]
            ):
                if (
                    self.config["model"]["variance_adaptor"]["variance_levels"][j]
                    == "phone"
                ):
                    mask = "src_mask"
                elif (
                    self.config["model"]["variance_adaptor"]["variance_levels"][j]
                    == "frame"
                ):
                    mask = "tgt_mask"
                if (
                    self.config["model"]["variance_adaptor"]["variance_transforms"][j]
                    == "cwt"
                ):
                    pred_dict["variances"][var] = {
                        "spectrogram": inference_result[f"variances_{var}"][
                            "spectrogram"
                        ][i][~inference_result[mask][i]].cpu()
                    }

                    pred_dict["variances"][var]["original_signal"] = inference_result[
                        f"variances_{var}"
                    ]["reconstructed_signal"][i][~inference_result[mask][i]].cpu()
                else:
                    pred_dict["variances"][var] = inference_result[f"variances_{var}"][
                        i
                    ][~inference_result[mask][i]].cpu()
            true_dict = {
                "mel": true_mel,
                "duration": batch["duration"][i][~result["src_mask"][i]].cpu(),
                "text": batch["text"][i],
                "raw_text": batch["raw_text"][i],
                "variances": {},
                "priors": {},
            }
            for j, var in enumerate(
                self.config["model"]["variance_adaptor"]["variances"]
            ):
                if (
                    self.config["model"]["variance_adaptor"]["variance_levels"][j]
                    == "phone"
                ):
                    mask = "src_mask"
                elif (
                    self.config["model"]["variance_adaptor"]["variance_levels"][j]
                    == "frame"
                ):
                    mask = "tgt_mask"
                if (
                    self.config["model"]["variance_adaptor"]["variance_transforms"][j]
                    == "cwt"
                ):
                    true_dict["variances"][var] = {
                        "spectrogram": batch[f"variances_{var}_spectrogram"][i][
                            ~result[mask][i]
                        ].cpu()
                    }

                    true_dict["variances"][var]["original_signal"] = batch[
                        f"variances_{var}_original_signal"
                    ][i][~result[mask][i]].cpu()
                else:
                    true_dict["variances"][var] = batch[f"variances_{var}"][i][
                        ~result[mask][i]
                    ].cpu()

            for prior in self.config["model"]["priors"]["prior_types"]:
                true_dict["priors"][prior] = batch[f"priors_{prior}"][i]
                pred_dict["priors"][prior] = batch[f"priors_{prior}"][i]

            if pred_dict["duration"].sum() == 0:
                print("WARNING: duration is zero (common at beginning of training)")
            else:
                try:
                    pred_fig = self.valid_ds.plot(pred_dict, show=False)
                    true_fig = self.valid_ds.plot(true_dict, show=False)
                    if self.valid_example_directory is not None:
                        Path(self.valid_example_directory).mkdir(
                            parents=True, exist_ok=True
                        )
                        pred_fig.save(
                            os.path.join(
                                self.valid_example_directory,
                                f"pred_{batch['id'][i]}.png",
                            )
                        )
                        true_fig.save(
                            os.path.join(
                                self.valid_example_directory,
                                f"true_{batch['id'][i]}.png",
                            )
                        )
                    pred_audio = self.vocoder(pred_mel.to(self.device).float())[
                        0
                    ]  # TODO: Implement predict_step
                    true_audio = self.vocoder(true_mel.to(self.device).float())[
                        0
                    ]  # TODO: Implement predict_step
                    self.eval_log_data.append(
                        [
                            batch["text"][i],
                            wandb.Image(pred_fig),
                            wandb.Image(true_fig),
                            wandb.Audio(pred_audio, sample_rate=22050),
                            wandb.Audio(true_audio, sample_rate=22050),
                        ]
                    )  # TODO: replace wandb with tensorboard
                except:
                    print(
                        "WARNING: failed to log example (common before training starts)"
                    )

    def _add_to_results_dict(self, inference_result, batch, result, add_n):
        # duration
        self.results_dict["duration"]["pred"] += list(
            inference_result["duration_rounded"][~inference_result["src_mask"]]
        )[:add_n]
        self.results_dict["duration"]["true"] += list(
            batch["duration"][~result["src_mask"]]
        )[:add_n]

        # mel
        self.results_dict["mel"]["pred"] += list(
            inference_result["mel"][~inference_result["tgt_mask"]]
        )[:add_n]
        self.results_dict["mel"]["true"] += list(batch["mel"][~result["tgt_mask"]])[
            :add_n
        ]

        for i, var in enumerate(self.config["model"]["variance_adaptor"]["variances"]):
            if (
                self.config["model"]["variance_adaptor"]["variance_levels"][i]
                == "phone"
            ):
                mask = "src_mask"
            elif (
                self.config["model"]["variance_adaptor"]["variance_levels"][i]
                == "frame"
            ):
                mask = "tgt_mask"
            if (
                self.config["model"]["variance_adaptor"]["variance_transforms"][i]
                == "cwt"
            ):
                self.results_dict[var]["pred"] += list(
                    inference_result[f"variances_{var}"]["reconstructed_signal"][
                        ~inference_result[mask]
                    ]
                )[:add_n]
                self.results_dict[var]["true"] += list(
                    batch[f"variances_{var}_original_signal"][~result[mask]]
                )[:add_n]
            else:
                self.results_dict[var]["pred"] += list(
                    inference_result[f"variances_{var}"][~inference_result[mask]]
                )[:add_n]
                self.results_dict[var]["true"] += list(
                    batch[f"variances_{var}"][~result[mask]]
                )[:add_n]

    def validation_epoch_end(self, validation_step_outputs):
        if self.trainer.is_global_zero:
            wandb.init(project="FastSpeech2")
            table = wandb.Table(
                data=self.eval_log_data,
                columns=[
                    "text",
                    "predicted_mel",
                    "true_mel",
                    "predicted_audio",
                    "true_audio",
                ],
            )
            wandb.log({"examples": table})
            if (
                not hasattr(self, "best_variances")
                and self.config["training"]["feature_prediction"]["early_stopping"][
                    "metric"
                ]
            ):
                self.best_variances = {}
            for key in self.results_dict.keys():
                self.results_dict[key]["pred"] = [
                    x.cpu().numpy() for x in self.results_dict[key]["pred"]
                ]
                self.results_dict[key]["true"] = [
                    x.cpu().numpy() for x in self.results_dict[key]["true"]
                ]
                if key not in "mel":
                    pred_list = np.random.choice(
                        np.array(self.results_dict[key]["pred"]), size=500
                    ).reshape(-1, 1)
                    true_list = np.random.choice(
                        np.array(self.results_dict[key]["true"]), size=500
                    ).reshape(-1, 1)
                    kde_pred = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
                        pred_list
                    )
                    kde_true = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
                        true_list
                    )
                    min_val = min(min(pred_list), min(true_list))
                    max_val = max(max(pred_list), max(true_list))
                    arange = np.arange(
                        min_val, max_val, (max_val - min_val) / 100
                    ).reshape(-1, 1)
                    var_js = jensenshannon(
                        np.exp(kde_pred.score_samples(arange)),
                        np.exp(kde_true.score_samples(arange)),
                    )
                    try:
                        var_mae = np.mean(
                            np.abs(
                                np.array(self.results_dict[key]["pred"])
                                - np.array(self.results_dict[key]["true"])
                            )
                        )
                    except:
                        self.eval_log_data = None
                        print(
                            "WARNING: failed to log validation results, this should be fixed in the next release"
                        )
                        return
                    if (
                        self.config["training"]["feature_prediction"]["early_stopping"][
                            "metric"
                        ]
                        != "none"
                        and not (
                            key in self.best_variances
                            and self.best_variances[key][1] == -1
                        )
                        and key != "duration"  # TODO: add duration to early stopping
                    ):
                        if key not in self.best_variances:
                            if (
                                self.config["training"]["feature_prediction"][
                                    "early_stopping"
                                ]["metric"]
                                == "mae"
                            ):
                                self.best_variances[key] = [var_mae, 1]
                            elif (
                                self.config["training"]["feature_prediction"][
                                    "early_stopping"
                                ]["metric"]
                                == "js"
                            ):
                                self.best_variances[key] = [var_js, 1]
                            torch.save(
                                self.variance_adaptor.encoders[key].state_dict(),
                                self.variance_encoder_dir / f"{key}_encoder_best.pt",
                            )
                        else:
                            if (
                                var_js < self.best_variances[key][0]
                                and self.config["training"]["feature_prediction"][
                                    "early_stopping"
                                ]["metric"]
                                == "js"
                            ):
                                self.best_variances[key] = [var_js, 1]
                                torch.save(
                                    self.variance_adaptor.encoders[key].state_dict(),
                                    self.variance_encoder_dir
                                    / f"{key}_encoder_best.pt",
                                )
                            elif (
                                var_mae < self.best_variances[key][0]
                                and self.config["training"]["feature_prediction"][
                                    "early_stopping"
                                ]["metric"]
                                == "mae"
                            ):
                                self.best_variances[key] = [var_mae, 1]
                                torch.save(
                                    self.variance_adaptor.encoders[key].state_dict(),
                                    self.variance_encoder_dir
                                    / f"{key}_encoder_best.pt",
                                )
                            else:
                                self.best_variances[key][1] += 1
                            if (
                                self.config["training"]["feature_prediction"][
                                    "early_stopping"
                                ]["patience"]
                                <= self.best_variances[key][1]
                            ):
                                self.best_variances[key][1] = -1
                                self.variance_adaptor.encoders[key].load_state_dict(
                                    torch.load(
                                        self.variance_encoder_dir
                                        / f"{key}_encoder_best.pt"
                                    )
                                )
                                # freeze encoder
                                print(f"Freezing encoder {key}")
                                self.variance_adaptor.freeze(key)
                                self.log_dict(
                                    {
                                        f"variance_early_stopping_{key}_epoch": self.current_epoch
                                    }
                                )

                    self.log_dict({f"eval/jensenshannon_{key}": var_js})
                    self.log_dict({f"eval/mae_{key}": var_mae})
                else:
                    pred_res = np.concatenate(
                        [
                            np.array([x[i] for x in self.results_dict[key]["pred"]])
                            for i in range(
                                self.config["preprocessing"]["audio"]["n_mels"]
                            )
                        ]
                    )
                    true_res = np.concatenate(
                        [
                            np.array([x[i] for x in self.results_dict[key]["true"]])
                            for i in range(
                                self.config["preprocessing"]["audio"]["n_mels"]
                            )
                        ]
                    )
                    pred_list = np.random.choice(pred_res, size=500).reshape(-1, 1)
                    true_list = np.random.choice(true_res, size=500).reshape(-1, 1)
                    kde_pred = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
                        pred_list
                    )
                    kde_true = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
                        true_list
                    )
                    min_val = min(min(pred_list), min(true_list))
                    max_val = max(max(pred_list), max(true_list))
                    arange = np.arange(
                        min_val, max_val, (max_val - min_val) / 100
                    ).reshape(-1, 1)
                    mel_js = jensenshannon(
                        np.exp(kde_pred.score_samples(arange)),
                        np.exp(kde_true.score_samples(arange)),
                    )
                    mel_softdtw_1 = SoftDTW(normalize=True, gamma=1)(
                        torch.tensor(self.results_dict[key]["pred"]).float(),
                        torch.tensor(self.results_dict[key]["true"]).float(),
                    )
                    mel_softdtw_0 = SoftDTW(normalize=True, gamma=0.001)(
                        torch.tensor(self.results_dict[key]["pred"]).float(),
                        torch.tensor(self.results_dict[key]["true"]).float(),
                    )
                    self.log_dict(
                        {
                            f"eval/jensenshannon_{key}": mel_js,
                            f"eval/softdtw_{key}": mel_softdtw_1,
                            f"eval/softdtw_{key}_gamma0": mel_softdtw_0,
                        }
                    )
            self.eval_log_data = None

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            self.config["training"]["feature_prediction"]["optimizer"]["lr"],
            betas=self.config["training"]["feature_prediction"]["optimizer"]["betas"],
            eps=self.config["training"]["feature_prediction"]["optimizer"]["eps"],
            weight_decay=self.config["training"]["feature_prediction"]["optimizer"][
                "weight_decay"
            ],
        )

        self.scheduler = NoamLR(
            self.optimizer,
            self.config["training"]["feature_prediction"]["optimizer"]["warmup_steps"],
        )

        sched = {
            "scheduler": self.scheduler,
            "interval": "step",
        }

        return [self.optimizer], [sched]
