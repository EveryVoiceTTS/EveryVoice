import itertools
import math
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from smts.config.base_config import BaseConfig
from smts.model.utils import create_depthwise_separable_convolution
from smts.utils import (
    dynamic_range_compression_torch,
    get_spectral_transform,
    plot_spectrogram,
)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
    elif classname.find("Sequential") != -1:
        for layer in m:
            layer.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(torch.nn.Module):
    def __init__(
        self, config, channels, kernel_size=3, dilation=(1, 3, 5), depthwise=False
    ):
        super(ResBlock1, self).__init__()
        self.config = config
        self.depthwise = depthwise
        if self.depthwise:
            self.convs1 = nn.ModuleList(
                [
                    create_depthwise_separable_convolution(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=x,
                        padding=get_padding(kernel_size, x),
                        weight_norm=True,
                    )
                    for x in dilation
                ]
            )
        else:
            self.convs1 = nn.ModuleList(
                [
                    weight_norm(
                        Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=x,
                            padding=get_padding(kernel_size, x),
                        )
                    )
                    for x in dilation
                ]
            )
        self.convs1.apply(init_weights)
        if self.depthwise:
            self.convs2 = nn.ModuleList(
                [
                    create_depthwise_separable_convolution(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                        weight_norm=True,
                    )
                    for _ in dilation
                ]
            )
        else:
            self.convs2 = nn.ModuleList(
                [
                    weight_norm(
                        Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=1,
                            padding=get_padding(kernel_size, 1),
                        )
                    )
                    for _ in dilation
                ]
            )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = self.config["model"]["vocoder"]["activation_function"](x)
            xt = c1(xt)
            xt = self.config["model"]["vocoder"]["activation_function"](x)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            if layer.__class__.__name__ == "Sequential":
                for sub_layer in layer:
                    remove_weight_norm(sub_layer)
            else:
                remove_weight_norm(layer)
        for layer in self.convs2:
            if layer.__class__.__name__ == "Sequential":
                for sub_layer in layer:
                    remove_weight_norm(sub_layer)
            else:
                remove_weight_norm(layer)


class ResBlock2(torch.nn.Module):
    def __init__(
        self,
        config: BaseConfig,
        channels,
        kernel_size=3,
        dilation=(1, 3),
        depthwise=False,
    ):
        super(ResBlock2, self).__init__()
        self.config = config
        self.depthwise = depthwise
        if self.depthwise:
            self.convs = nn.ModuleList(
                [
                    create_depthwise_separable_convolution(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                        weight_norm=True,
                    ),
                    create_depthwise_separable_convolution(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                        weight_norm=True,
                    ),
                ]
            )
        else:
            self.convs = nn.ModuleList(
                [
                    weight_norm(
                        Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=dilation[0],
                            padding=get_padding(kernel_size, dilation[0]),
                        )
                    ),
                    weight_norm(
                        Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=dilation[1],
                            padding=get_padding(kernel_size, dilation[1]),
                        )
                    ),
                ]
            )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = self.config["model"]["vocoder"]["activation_function"](x)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            if layer.__class__.__name__ == "Sequential":
                for sub_layer in layer:
                    remove_weight_norm(sub_layer)
            else:
                remove_weight_norm(layer)


class Generator(torch.nn.Module):
    def __init__(self, config: BaseConfig):
        super(Generator, self).__init__()
        self.config = config
        self.depthwise = config["model"]["vocoder"]["depthwise_separable_convolutions"][
            "generator"
        ]
        self.audio_config = config["preprocessing"]["audio"]
        self.sampling_rate_change = (
            self.audio_config["output_sampling_rate"]
            // self.audio_config["input_sampling_rate"]
        )
        self.model_vocoder_config = config["model"]["vocoder"]
        self.num_kernels = len(self.model_vocoder_config["resblock_kernel_sizes"])
        self.num_upsamples = len(self.model_vocoder_config["upsample_rates"])
        self.conv_pre = (
            create_depthwise_separable_convolution(
                in_channels=self.audio_config["n_mels"],
                out_channels=self.model_vocoder_config["upsample_initial_channel"],
                kernel_size=7,
                stride=1,
                padding=3,
                weight_norm=True,
            )
            if self.depthwise
            else weight_norm(
                Conv1d(
                    self.audio_config["n_mels"],
                    self.model_vocoder_config["upsample_initial_channel"],
                    7,
                    1,
                    padding=3,
                )
            )
        )  # in  # out  # kernel_size  # stride

        resblock = (
            ResBlock1 if self.model_vocoder_config["resblock"] == "1" else ResBlock2
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(
                self.model_vocoder_config["upsample_rates"],
                self.model_vocoder_config["upsample_kernel_sizes"],
            )
        ):
            # TODO: add sensible upsampling layer
            if self.depthwise:
                self.ups.append(
                    create_depthwise_separable_convolution(
                        in_channels=self.model_vocoder_config[
                            "upsample_initial_channel"
                        ]
                        // (2**i),
                        out_channels=self.model_vocoder_config[
                            "upsample_initial_channel"
                        ]
                        // (2 ** (i + 1)),
                        kernel_size=k,
                        stride=u,
                        padding=(k - u) // 2,
                        transpose=True,
                        weight_norm=True,
                    )
                )
            else:
                self.ups.append(
                    weight_norm(
                        ConvTranspose1d(
                            self.model_vocoder_config["upsample_initial_channel"]
                            // (2**i),  # in
                            self.model_vocoder_config["upsample_initial_channel"]
                            // (2 ** (i + 1)),  # out
                            k,  # kernel
                            u,  # stride
                            padding=(k - u) // 2,
                        )
                    )
                )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = self.model_vocoder_config["upsample_initial_channel"] // (2 ** (i + 1))
            for k, d in zip(
                self.model_vocoder_config["resblock_kernel_sizes"],
                self.model_vocoder_config["resblock_dilation_sizes"],
            ):
                self.resblocks.append(
                    resblock(self.config, ch, k, d, depthwise=self.depthwise)
                )
        if self.config["model"]["vocoder"]["istft_layer"]:
            self.post_n_fft = (
                self.audio_config["n_fft"] * self.sampling_rate_change
            ) // math.prod(self.config["model"]["vocoder"]["upsample_rates"])
            self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
            conv_post_out_channels = self.post_n_fft + 2
        else:
            self.post_n_fft = conv_post_out_channels = 1

        if self.depthwise:
            self.conv_post = create_depthwise_separable_convolution(
                ch, conv_post_out_channels, 7, 1, padding=3, weight_norm=True
            )
        else:
            self.conv_post = weight_norm(
                Conv1d(ch, conv_post_out_channels, 7, 1, padding=3)
            )
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.config["model"]["vocoder"]["activation_function"](x)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.config["model"]["vocoder"]["activation_function"](x)
        if self.config["model"]["vocoder"]["istft_layer"]:
            x = self.reflection_pad(x)
            x = self.conv_post(x)
            spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
            phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
            return spec, phase
        else:
            x = self.conv_post(x)
            x = torch.tanh(x)
            return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for layer in self.ups:
            if layer.__class__.__name__ == "Sequential":
                for sub_layer in layer:
                    remove_weight_norm(sub_layer)
            else:
                remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        if self.conv_pre.__class__.__name__ == "Sequential":
            for sub_layer in self.conv_pre:
                remove_weight_norm(sub_layer)
        else:
            remove_weight_norm(self.conv_pre)
        if self.conv_post.__class__.__name__ == "Sequential":
            for sub_layer in self.conv_post:
                remove_weight_norm(sub_layer)
        else:
            remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(
        self, period, config, kernel_size=5, stride=3, use_spectral_norm=False
    ):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.config = config
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = self.config["model"]["vocoder"]["activation_function"](x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, config):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2, config),
                DiscriminatorP(3, config),
                DiscriminatorP(5, config),
                DiscriminatorP(7, config),
                DiscriminatorP(11, config),
            ]
        )

    def forward_interpolates(self, interp):
        y_ds = []
        f_maps = []
        for d in self.discriminators:
            y_d, f_map = d(interp)
            y_ds.append(y_d)
            f_maps.append(f_map)
        return y_ds, f_maps

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, config, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        self.config = config
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = self.config["model"]["vocoder"]["activation_function"](x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, config):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(config, use_spectral_norm=True),
                DiscriminatorS(config),
                DiscriminatorS(config),
            ]
        )
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)]
        )

    def forward_interpolates(self, interp):
        y_ds = []
        f_maps = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                interp = self.meanpools[i - 1](interp)
            y_d, f_map = d(interp)
            y_ds.append(y_d)
            f_maps.append(f_map)
        return y_ds, f_maps

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class HiFiGAN(pl.LightningModule):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self.mpd = MultiPeriodDiscriminator(config)
        self.msd = MultiScaleDiscriminator(config)
        self.generator = Generator(config)
        self.batch_size = self.config["training"][
            "batch_size"
        ]  # this is declared explicitly so that auto_scale_batch_size works: https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html
        self.save_hyperparameters()
        self.audio_config = config["preprocessing"]["audio"]
        self.sampling_rate_change = (
            self.audio_config["output_sampling_rate"]
            // self.audio_config["input_sampling_rate"]
        )
        self.use_gradient_penalty = (
            self.config["training"]["vocoder"]["gan_type"] == "wgan-gp"
        )
        self.use_wgan = self.config["training"]["vocoder"]["gan_type"] == "wgan"
        # We don't have to set the fft size and hop/window lengths as hyperparameters here, because we can just multiply by the upsampling rate
        self.spectral_transform = get_spectral_transform(
            self.audio_config["spec_type"],
            self.audio_config["n_fft"] * self.sampling_rate_change,
            self.audio_config["fft_window_frames"] * self.sampling_rate_change,
            self.audio_config["fft_hop_frames"] * self.sampling_rate_change,
            f_min=self.audio_config["f_min"],
            f_max=self.audio_config["f_max"],
            sample_rate=self.audio_config["output_sampling_rate"],
            n_mels=self.audio_config["n_mels"],
        )
        if self.config["model"]["vocoder"]["istft_layer"]:
            self.inverse_spectral_transform = get_spectral_transform(
                "istft",
                self.generator.post_n_fft,
                self.generator.post_n_fft,
                self.generator.post_n_fft // 4,
            )
        # TODO: figure out multiple nodes/gpus: https://pytorch-lightning.readthedocs.io/en/1.4.0/advanced/multi_gpu.html
        # TODO: figure out freezing layers

    def forward(self, x):
        return self.generator(x)

    def on_save_checkpoint(self, checkpoint):
        version = (
            self.config["training"]["logger"]["version"]
            or f"version_{self.logger.version}"
        )
        gen_dir = os.path.join(
            self.config["training"]["logger"]["save_dir"],
            self.config["training"]["logger"]["name"],
            version,
            "checkpoints",
        )
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)
        gen_step = self.global_step // 2
        gen_path = os.path.join(
            self.config["training"]["logger"]["save_dir"],
            self.config["training"]["logger"]["name"],
            version,
            "checkpoints",
            f"g{gen_step}.ckpt",
        )
        torch.save(
            self.generator.state_dict(),
            gen_path,
        )
        return checkpoint

    def configure_optimizers(self):
        if self.config["training"]["vocoder"]["gan_optimizer"] == "adam":
            optim_g = torch.optim.AdamW(
                self.generator.parameters(),
                self.config["training"]["vocoder"]["learning_rate"],
                betas=[
                    self.config["training"]["vocoder"]["adam_b1"],
                    self.config["training"]["vocoder"]["adam_b2"],
                ],
            )
            optim_d = torch.optim.AdamW(
                itertools.chain(self.msd.parameters(), self.mpd.parameters()),
                self.config["training"]["vocoder"]["learning_rate"],
                betas=[
                    self.config["training"]["vocoder"]["adam_b1"],
                    self.config["training"]["vocoder"]["adam_b2"],
                ],
            )
        else:
            optim_g = torch.optim.RMSprop(
                self.generator.parameters(),
                lr=self.config["training"]["vocoder"]["learning_rate"],
            )
            optim_d = torch.optim.RMSprop(
                itertools.chain(self.msd.parameters(), self.mpd.parameters()),
                lr=self.config["training"]["vocoder"]["learning_rate"],
            )
        if self.use_wgan or self.use_gradient_penalty:
            return (
                {"optimizer": optim_g, "frequency": 1},
                {"optimizer": optim_d, "frequency": 5},
            )
        else:
            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
                optim_g, gamma=self.config["training"]["vocoder"]["lr_decay"]
            )
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
                optim_d, gamma=self.config["training"]["vocoder"]["lr_decay"]
            )
            return [optim_g, optim_d], [scheduler_g, scheduler_d]

    def feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2

    def discriminator_loss(
        self,
        disc_real_outputs,
        disc_generated_outputs,
        gp=None,
        lambda_gp=10,
    ):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            if gp is not None:
                r_loss = -torch.mean(dr)
                g_loss = torch.mean(dg)
                loss += r_loss + g_loss + lambda_gp * gp
            else:
                r_loss = torch.mean((1 - dr) ** 2)
                g_loss = torch.mean(dg**2)
                loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses

    def generator_loss(self, disc_outputs, gp=True):
        g_loss = 0
        gen_losses = []
        for dg in disc_outputs:
            if gp:
                loss = -torch.mean(dg)
            else:
                loss = torch.mean((1 - dg) ** 2)
            gen_losses.append(loss)
            g_loss += loss

        return (g_loss, gen_losses)

    def compute_gradient_penalty(self, real_samples, fake_samples, discriminator):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        # real_samples.size() = (batch_size, 1, segment_size)
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1))).to(
            self.device
        )  # size = (batch_size, 1, 1)
        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(
            True
        )  # size = (batch_size, 1, segment_size)
        interpolates = interpolates.to(self.device)
        d_interpolates, _ = discriminator.forward_interpolates(interpolates)
        d_interpolates = [torch.mean(x, dim=1) for x in d_interpolates]
        d_interpolates = (
            torch.stack(d_interpolates).sum(dim=0).unsqueeze(1)
        )  # size = (batch_size, 1)
        fake = (
            torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        )  # size = (batch_size, 1)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[
            0
        ]  # size = (batch_size, 1, segment_size)
        gradients = gradients.view(gradients.size(0), -1).to(
            self.device
        )  # size = (batch_size, segment_size)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()  # size = (1)
        self.log("train/gradient_penalty", gradient_penalty)
        return gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y, _, y_mel = batch
        y = y.unsqueeze(1)
        # x.size() & y_mel.size() = [batch_size, n_mels=80, n_frames=32]
        # y.size() = [batch_size, segment_size=8192]
        # train generator
        if optimizer_idx == 0:
            # generate waveform
            if self.config["model"]["vocoder"]["istft_layer"]:
                mag, phase = self(x)
                self.generated_wav = self.inverse_spectral_transform(
                    mag * torch.exp(phase * 1j)
                ).unsqueeze(-2)
            else:
                self.generated_wav = self(x)
            # create mel
            generated_mel_spec = dynamic_range_compression_torch(
                self.spectral_transform(self.generated_wav).squeeze(1)[:, :, 1:]
            )
            # calculate loss
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, self.generated_wav)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, self.generated_wav)
            loss_fm_f = self.feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = self.feature_loss(fmap_s_r, fmap_s_g)
            # loss_gen_f = -torch.mean(y_df_hat_g)
            # loss_gen_s = -torch.mean(y_ds_hat_g)
            loss_gen_f, _ = self.generator_loss(
                y_df_hat_g, gp=self.use_gradient_penalty
            )
            loss_gen_s, _ = self.generator_loss(
                y_ds_hat_g, gp=self.use_gradient_penalty
            )
            self.log("training/gen/loss_fmap_f", loss_fm_f, prog_bar=False)
            self.log("training/gen/loss_fmap_s", loss_fm_s, prog_bar=False)
            self.log("training/gen/loss_gen_f", loss_gen_f, prog_bar=False)
            self.log("training/gen/loss_gen_s", loss_gen_s, prog_bar=False)
            loss_mel = F.l1_loss(y_mel, generated_mel_spec) * 45
            gen_loss_total = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            # log generator loss
            self.log("training/gen/gen_loss_total", gen_loss_total, prog_bar=False)
            self.log("training/gen/mel_spec_error", loss_mel / 45, prog_bar=False)
            return gen_loss_total

        # train discriminators
        if (
            self.global_step >= self.config["training"]["vocoder"]["generator_warmup"]
            and optimizer_idx == 1
        ):
            if self.config["model"]["vocoder"]["istft_layer"]:
                mag, phase = self(x)
                y_g_hat = (
                    self.inverse_spectral_transform(mag * torch.exp(phase * 1j))
                    .unsqueeze(-2)
                    .detach()
                )
            else:
                y_g_hat = self(x).detach()
            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat)
            if self.use_gradient_penalty:
                gp_f = self.compute_gradient_penalty(y.data, y_g_hat.data, self.mpd)
            else:
                gp_f = None
            loss_disc_f, _, _ = self.discriminator_loss(y_df_hat_r, y_df_hat_g, gp=gp_f)
            self.log("training/disc/mpd_loss", loss_disc_f, prog_bar=False)
            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat)
            # gp_s = self.compute_gradient_penalty(y_ds_hat_r, y_ds_hat_g)
            # loss_disc_s = -torch.mean(y_ds_hat_r) + torch.mean(y_ds_hat_g) + 10 * gp_s
            if self.use_gradient_penalty:
                gp_s = self.compute_gradient_penalty(y.data, y_g_hat.data, self.msd)
            else:
                gp_s = None
            loss_disc_s, _, _ = self.discriminator_loss(y_ds_hat_r, y_ds_hat_g, gp=gp_s)
            self.log("training/disc/msd_loss", loss_disc_s, prog_bar=False)
            # WGAN
            if self.use_wgan:
                for p in self.msd.parameters():
                    p.data.clamp_(
                        -self.config["training"]["vocoder"]["wgan_clip_value"],
                        self.config["training"]["vocoder"]["wgan_clip_value"],
                    )
                for p in self.mpd.parameters():
                    p.data.clamp_(
                        -self.config["training"]["vocoder"]["wgan_clip_value"],
                        self.config["training"]["vocoder"]["wgan_clip_value"],
                    )
            # calculate loss
            disc_loss_total = loss_disc_s + loss_disc_f
            # log discriminator loss
            self.log("training/disc/d_loss_total", disc_loss_total, prog_bar=False)
            return disc_loss_total

    def validation_step(self, batch, batch_idx):
        # TODO: batch size should be 1, should process full files and samples should be selected chosen on the fly and not cached. Look into DistributedSampler from HiFiGAN
        x, y, bn, y_mel = batch
        current_step = (
            self.global_step // 2
        )  # because self.global_step counts gen/disc steps
        # generate waveform
        if self.config["model"]["vocoder"]["istft_layer"]:
            mag, phase = self(x)
            self.generated_wav = self.inverse_spectral_transform(
                mag * torch.exp(phase * 1j)
            ).unsqueeze(-2)
        else:
            self.generated_wav = self(x)
        # create mel
        generated_mel_spec = dynamic_range_compression_torch(
            self.spectral_transform(self.generated_wav).squeeze(1)[:, :, 1:]
        )
        val_err_tot = F.l1_loss(y_mel, generated_mel_spec).item()
        # # Below is taken from HiFiGAN
        if self.global_step == 0:
            # Log ground truth audio and spec
            self.logger.experiment.add_audio(
                f"gt/y_{bn[0]}",
                y[0],
                current_step,
                self.config["preprocessing"]["audio"]["output_sampling_rate"],
            )
            self.logger.experiment.add_figure(
                f"gt/y_spec_{bn[0]}",
                plot_spectrogram(x[0].cpu().numpy()),
                current_step,
            )
        #
        if batch_idx == 0:
            self.logger.experiment.add_audio(
                f"generated/y_hat_{bn[0]}",
                self.generated_wav[0],
                current_step,
                self.config["preprocessing"]["audio"]["output_sampling_rate"],
            )

            y_hat_spec = dynamic_range_compression_torch(
                self.spectral_transform(self.generated_wav[0]).squeeze(1)[:, :, 1:]
            )
            self.logger.experiment.add_figure(
                f"generated/y_hat_spec_{bn[0]}",
                plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()),
                current_step,
            )

        self.log("validation/mel_spec_error", val_err_tot, prog_bar=False)
