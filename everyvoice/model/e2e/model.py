import json

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger

from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import FastSpeech2
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions import (
    Stats,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.utils import plot_mel
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import HiFiGAN
from everyvoice.utils.heavy import dynamic_range_compression_torch, expand


class EveryVoice(pl.LightningModule):
    def __init__(self, config: EveryVoiceConfig):
        super().__init__()
        self.config = config
        self.batch_size = config.training.batch_size
        if self.config.training.feature_prediction_checkpoint is not None:
            logger.info(
                f"Loading FastSpeech2 model from checkpoint {self.config.training.feature_prediction_checkpoint}"
            )
            self.feature_prediction: FastSpeech2 = FastSpeech2.load_from_checkpoint(
                self.config.training.feature_prediction_checkpoint
            )
            self.feature_prediction.config = self.config.feature_prediction
        else:
            self.feature_prediction = FastSpeech2(config.feature_prediction)
        if self.config.training.vocoder_checkpoint is not None:
            logger.info(
                f"Loading HiFiGAN model from checkpoint {self.config.training.vocoder_checkpoint}"
            )
            self.vocoder: HiFiGAN = HiFiGAN.load_from_checkpoint(
                self.config.training.vocoder_checkpoint
            )
            self.vocoder.config = self.config.vocoder
        else:
            self.vocoder = HiFiGAN(config.vocoder)
        self.sampling_rate_change = (
            self.config.vocoder.preprocessing.audio.output_sampling_rate
            // self.config.vocoder.preprocessing.audio.input_sampling_rate
        )
        with open(
            self.config.feature_prediction.preprocessing.save_dir / "stats.json"
        ) as f:
            self.stats: Stats = Stats(**json.load(f))
        self.input_frame_segment_size = (
            self.config.vocoder.preprocessing.audio.vocoder_segment_size
            // self.config.vocoder.preprocessing.audio.fft_hop_frames
        )
        self.use_segments = True

    def forward(self, batch, inference=False):
        feature_prediction = self.feature_prediction.forward(batch)
        vocoder_x = feature_prediction["postnet_output"]
        if not inference and self.use_segments:
            vocoder_x = torch.stack(
                [
                    x[
                        batch["segment_start_frame"][i] : batch["segment_start_frame"][
                            i
                        ]
                        + self.input_frame_segment_size
                    ]
                    for i, x in enumerate(vocoder_x)
                ]
            )
        # Generate Waveform
        if self.config.vocoder.model.istft_layer:
            mag, phase = self.vocoder(vocoder_x.transpose(1, 2))
            vocoder_prediction = self.vocoder.inverse_spectral_transform(
                mag * torch.exp(phase * 1j)
            ).unsqueeze(-2)
        else:
            vocoder_prediction = self.vocoder(vocoder_x.transpose(1, 2))
        return feature_prediction, vocoder_prediction

    def vocoder_loss(self, y, y_mel, y_hat, y_hat_mel, mode="training"):
        _, y_df_hat_g, fmap_f_r, fmap_f_g = self.vocoder.mpd(y, y_hat)
        _, y_ds_hat_g, fmap_s_r, fmap_s_g = self.vocoder.msd(y, y_hat)
        loss_fm_f = self.vocoder.feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = self.vocoder.feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, _ = self.vocoder.generator_loss(
            y_df_hat_g, gp=self.vocoder.use_gradient_penalty
        )
        loss_gen_s, _ = self.vocoder.generator_loss(
            y_ds_hat_g, gp=self.vocoder.use_gradient_penalty
        )
        loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * 45
        gen_loss_total = 0
        # Log Losses
        self.log(f"{mode}/vocoder/gen/loss_fmap_f", loss_fm_f, prog_bar=False)
        self.log(f"{mode}/vocoder/gen/loss_fmap_s", loss_fm_s, prog_bar=False)
        self.log(f"{mode}/vocoder/gen/loss_gen_f", loss_gen_f, prog_bar=False)
        self.log(f"{mode}/vocoder/gen/loss_gen_s", loss_gen_s, prog_bar=False)
        self.log(f"{mode}/vocoder/gen/mel_spec_error", loss_mel / 45, prog_bar=False)
        gen_loss_total = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
        self.log(f"{mode}/vocoder/gen/gen_loss_total", gen_loss_total, prog_bar=False)
        return gen_loss_total

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Train Feature Prediction & Vocoder Generator
        if optimizer_idx == 0:
            # Generate Predictions
            fp_output, vocoder_y_hat = self(batch)
            # Calculate Feature Prediction Loss
            fp_losses = self.feature_prediction.loss(fp_output, batch)
            self.log_dict(
                {
                    f"training/feature_prediction/{k}_loss": v.item()
                    for k, v in fp_losses.items()
                }
            )
            # Create Mel of generated wav
            generated_mel_spec = dynamic_range_compression_torch(
                self.vocoder.spectral_transform(vocoder_y_hat).squeeze(1)[:, :, 1:]
            )
            # Calculate Generator Loss
            vocoder_gen_loss_total = self.vocoder_loss(
                batch["audio"], batch["audio_mel"], vocoder_y_hat, generated_mel_spec
            )
            return fp_losses["total"] + vocoder_gen_loss_total

        # Train Vocoder Discriminator
        if (
            self.global_step >= self.vocoder.config.training.generator_warmup_steps
            and optimizer_idx == 1
        ):
            with torch.no_grad():
                fp_output, y_g_hat = self(batch)
            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = self.vocoder.mpd(batch["audio"], y_g_hat)
            if self.vocoder.use_gradient_penalty:
                gp_f = self.vocoder.compute_gradient_penalty(
                    batch["audio"].data, y_g_hat.data, self.vocoder.mpd
                )
            else:
                gp_f = None
            loss_disc_f, _, _ = self.vocoder.discriminator_loss(
                y_df_hat_r, y_df_hat_g, gp=gp_f
            )
            self.log("training/disc/mpd_loss", loss_disc_f, prog_bar=False)
            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = self.vocoder.msd(batch["audio"], y_g_hat)
            # gp_s = self.compute_gradient_penalty(y_ds_hat_r, y_ds_hat_g)
            # loss_disc_s = -torch.mean(y_ds_hat_r) + torch.mean(y_ds_hat_g) + 10 * gp_s
            if self.vocoder.use_gradient_penalty:
                gp_s = self.vocoder.compute_gradient_penalty(
                    batch["audio"].data, y_g_hat.data, self.vocoder.msd
                )
            else:
                gp_s = None
            loss_disc_s, _, _ = self.vocoder.discriminator_loss(
                y_ds_hat_r, y_ds_hat_g, gp=gp_s
            )
            self.log("training/disc/msd_loss", loss_disc_s, prog_bar=False)
            # WGAN
            if self.vocoder.use_wgan:
                for p in self.vocoder.msd.parameters():
                    p.data.clamp_(
                        -self.config.vocoder.training.wgan_clip_value,
                        self.config.vocoder.training.wgan_clip_value,
                    )
                for p in self.vocoder.mpd.parameters():
                    p.data.clamp_(
                        -self.config.vocoder.training.wgan_clip_value,
                        self.config.vocoder.training.wgan_clip_value,
                    )
            # calculate loss
            disc_loss_total = loss_disc_s + loss_disc_f
            # log discriminator loss
            self.log("training/disc/d_loss_total", disc_loss_total, prog_bar=False)
            return disc_loss_total

    def validation_step(self, batch, batch_idx):
        fp_output, vocoder_y_hat = self(batch)
        fp_losses = self.feature_prediction.loss(fp_output, batch)
        generated_mel_spec = dynamic_range_compression_torch(
            self.vocoder.spectral_transform(vocoder_y_hat).squeeze(1)[:, :, 1:]
        )
        self.log_dict(
            {
                f"validation/feature_prediction/{k}_loss": v.item()
                for k, v in fp_losses.items()
            }
        )
        val_err_tot = torch.nn.functional.l1_loss(
            batch["audio_mel"], generated_mel_spec
        ).item()
        self.log("validation/mel_spec_error", val_err_tot, prog_bar=False)
        if self.global_step == 0:
            audio = torch.load(
                self.config.feature_prediction.preprocessing.save_dir
                / "audio"
                / self.config.feature_prediction.preprocessing.value_separator.join(
                    [
                        batch["basename"][0],
                        batch["speaker"][0],
                        batch["language"][0],
                        f"audio-{self.config.feature_prediction.preprocessing.audio.input_sampling_rate}.pt",
                    ]
                )
            )
            # Log ground truth audio
            self.logger.experiment.add_audio(
                f"gt/wav_{batch['basename'][0]}",
                audio,
                self.global_step,
                self.config.feature_prediction.preprocessing.audio.output_sampling_rate,
            )
        if batch_idx == 0:
            fp_output, synthesis_wav_output = self(batch, inference=True)
            duration_np = batch["duration"][0].cpu().numpy()
            self.logger.experiment.add_figure(
                f"pred/spec_{batch['basename'][0]}",
                plot_mel(
                    [
                        {
                            "mel": np.swapaxes(batch["mel"][0].cpu().numpy(), 0, 1),
                            "pitch": expand(
                                batch["pitch"][0].cpu().numpy(), duration_np
                            ),
                            "energy": expand(
                                batch["energy"][0].cpu().numpy(), duration_np
                            ),
                        },
                        {
                            "mel": np.swapaxes(
                                fp_output["postnet_output"][0].cpu().numpy(), 0, 1
                            ),
                            "pitch": expand(
                                fp_output["pitch_prediction"][0].cpu().numpy(),
                                duration_np,
                            ),
                            "energy": expand(
                                fp_output["energy_prediction"][0].cpu().numpy(),
                                duration_np,
                            ),
                        },
                    ],
                    self.stats,
                    ["Ground-Truth Spectrogram", "Synthesized Spectrogram"],
                ),
                self.global_step,
            )
            self.logger.experiment.add_audio(
                f"pred/wav_{batch['basename'][0]}",
                synthesis_wav_output,
                self.global_step,
                self.config.vocoder.preprocessing.audio.output_sampling_rate,
            )
            if self.config.vocoder.model.istft_layer:
                mag, phase = self.vocoder(batch["mel"].transpose(1, 2))
                copy_synthesis = self.vocoder.inverse_spectral_transform(
                    mag * torch.exp(phase * 1j)
                ).unsqueeze(-2)
            else:
                copy_synthesis = self.vocoder(batch["mel"].transpose(1, 2))
            self.logger.experiment.add_audio(
                f"copy-synthesis/wav_{batch['basename'][0]}",
                copy_synthesis,
                self.global_step,
                self.config.vocoder.preprocessing.audio.output_sampling_rate,
            )
        return fp_losses["total"] + val_err_tot

    def configure_optimizers(self):
        return self.vocoder.configure_optimizers()
