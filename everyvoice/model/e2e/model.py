import json

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
from loguru import logger

from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
    FastSpeech2,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions_heavy import (
    Stats,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.utils.heavy import (
    plot_mel,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import HiFiGAN
from everyvoice.text.lookups import lookuptables_from_config
from everyvoice.utils.heavy import dynamic_range_compression_torch, expand


class EveryVoice(pl.LightningModule):
    def __init__(self, config: EveryVoiceConfig):
        super().__init__()
        # Required by PyTorch Lightning for our manual optimization
        self.automatic_optimization = False
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
            lang2id, speaker2id = lookuptables_from_config(config.feature_prediction)
            # TODO: What about when we are fine-tuning? Do the bins in the Variance Adaptor not change? https://github.com/roedoejet/FastSpeech2_lightning/issues/28
            with open(
                config.feature_prediction.preprocessing.save_dir / "stats.json"
            ) as f:
                stats: Stats = Stats(**json.load(f))
            self.feature_prediction = FastSpeech2(
                config.feature_prediction,
                stats=stats,
                lang2id=lang2id,
                speaker2id=speaker2id,
            )
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
        # Required by PyTorch Lightning for our manual optimization
        self.vocoder.automatic_optimization = False
        self.feature_prediction.automatic_optimization = False
        self.sampling_rate_change = (
            self.config.vocoder.preprocessing.audio.output_sampling_rate
            // self.config.vocoder.preprocessing.audio.input_sampling_rate
        )
        self.input_frame_segment_size = (
            self.config.vocoder.preprocessing.audio.vocoder_segment_size
            // self.config.vocoder.preprocessing.audio.fft_hop_size
        )
        self.use_segments = True

    def forward(self, batch, inference=False):
        feature_prediction = self.feature_prediction.forward(batch)
        vocoder_x = feature_prediction[self.feature_prediction.output_key]
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
        loss_gen_f, _ = self.vocoder.generator_loss(y_df_hat_g)
        loss_gen_s, _ = self.vocoder.generator_loss(y_ds_hat_g)
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

    def training_step(self, batch, batch_idx):
        y = batch["audio"]
        y_mel = batch["audio_mel"]
        # Train Feature Prediction & Vocoder Generator
        optim_fp, optim_g, optim_d = self.optimizers()
        scheduler_fp, scheduler_g, scheduler_d = self.lr_schedulers()
        # Generate Predictions
        fp_output, generated_wav = self(batch)
        # Create Mel of generated wav
        generated_mel_spec = dynamic_range_compression_torch(
            self.vocoder.spectral_transform(generated_wav).squeeze(1)[:, :, 1:]
        )

        # Train Vocoder Discriminator
        if self.global_step >= self.vocoder.config.training.generator_warmup_steps:
            optim_d.zero_grad()
            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = self.vocoder.mpd(y, generated_wav.detach())
            loss_disc_f, _, _ = self.vocoder.discriminator_loss(y_df_hat_r, y_df_hat_g)
            self.log("training/disc/mpd_loss", loss_disc_f, prog_bar=False)
            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = self.vocoder.msd(y, generated_wav.detach())
            loss_disc_s, _, _ = self.vocoder.discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            self.log("training/disc/msd_loss", loss_disc_s, prog_bar=False)
            # WGAN
            if self.vocoder.use_wgan:
                for p in self.vocoder.msd.parameters():
                    p.data.clamp_(
                        -self.config.training.wgan_clip_value,
                        self.config.training.wgan_clip_value,
                    )
                for p in self.vocoder.mpd.parameters():
                    p.data.clamp_(
                        -self.config.training.wgan_clip_value,
                        self.config.training.wgan_clip_value,
                    )
            # calculate loss
            disc_loss_total = loss_disc_s + loss_disc_f
            # manual optimization because Pytorch Lightning 2.0+ doesn't handle automatic optimization for multiple optimizers
            # use .backward for now, but maybe switch to self.manual_backward() in the future: https://github.com/Lightning-AI/lightning/issues/18740
            disc_loss_total.backward(retain_graph=True)
            # clip gradients
            self.clip_gradients(
                optim_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
            )
            optim_d.step()
            # step in the scheduler every epoch
            if self.trainer.is_last_batch:
                scheduler_d.step()
            # log discriminator loss
            self.log("training/disc/d_loss_total", disc_loss_total, prog_bar=False)
        # train generator
        optim_fp.zero_grad()
        optim_g.zero_grad()
        # Calculate Feature Prediction Loss
        fp_losses: torch.Tensor = self.feature_prediction.loss(
            fp_output, batch, self.current_epoch
        )
        self.log_dict(
            {
                f"training/feature_prediction/{k}_loss": v.item()
                for k, v in fp_losses.items()
            }
        )

        # Calculate Generator Loss
        loss_mel = F.l1_loss(y_mel, generated_mel_spec) * 45
        if self.global_step >= self.vocoder.config.training.generator_warmup_steps:
            _, y_df_hat_g, fmap_f_r, fmap_f_g = self.vocoder.mpd(y, generated_wav)
            _, y_ds_hat_g, fmap_s_r, fmap_s_g = self.vocoder.msd(y, generated_wav)
            loss_fm_f = self.vocoder.feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = self.vocoder.feature_loss(fmap_s_r, fmap_s_g)
            # loss_gen_f = -torch.mean(y_df_hat_g)
            # loss_gen_s = -torch.mean(y_ds_hat_g)
            loss_gen_f, _ = self.vocoder.generator_loss(y_df_hat_g)
            loss_gen_s, _ = self.vocoder.generator_loss(y_ds_hat_g)
            self.log("training/gen/loss_fmap_f", loss_fm_f, prog_bar=False)
            self.log("training/gen/loss_fmap_s", loss_fm_s, prog_bar=False)
            self.log("training/gen/loss_gen_f", loss_gen_f, prog_bar=False)
            self.log("training/gen/loss_gen_s", loss_gen_s, prog_bar=False)
            gen_loss_total = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
        else:
            gen_loss_total = loss_mel
        # manual optimization because Pytorch Lightning 2.0+ doesn't handle automatic optimization for multiple optimizers
        # use .backward for now, but maybe switch to self.manual_backward() in the future: https://github.com/Lightning-AI/lightning/issues/18740
        # self.manual_backward(gen_loss_total)
        gen_loss_total.backward(retain_graph=True)
        # clip gradients
        self.clip_gradients(
            optim_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        # fp_losses["total"].backward()
        optim_g.step()
        optim_fp.step()
        # step in the scheduler every epoch
        if self.trainer.is_last_batch:
            scheduler_g.step()
            scheduler_fp.step()
            # log generator loss
        self.log("training/gen/gen_loss_total", gen_loss_total, prog_bar=False)
        self.log("training/gen/mel_spec_error", loss_mel / 45, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        fp_output, vocoder_y_hat = self(batch)
        fp_losses = self.feature_prediction.loss(fp_output, batch, self.current_epoch)
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
            audio, _ = torchaudio.load(
                self.config.feature_prediction.preprocessing.save_dir
                / "audio"
                / "--".join(
                    [
                        batch["basename"][0],
                        batch["speaker"][0],
                        batch["language"][0],
                        f"audio-{self.config.feature_prediction.preprocessing.audio.input_sampling_rate}.wav",
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
            duration_np = fp_output["duration_target"][0].cpu().numpy()
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
                                fp_output[self.feature_prediction.output_key][0]
                                .cpu()
                                .numpy(),
                                0,
                                1,
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
                    self.feature_prediction.stats,
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
        fp_optims, fp_schedulers = self.feature_prediction.configure_optimizers()
        vocoder_optims, vocoder_schedulers = self.vocoder.configure_optimizers()
        return fp_optims + vocoder_optims, fp_schedulers + vocoder_schedulers
