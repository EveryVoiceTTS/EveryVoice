import pytorch_lightning as pl
import torch
from attn.attention import MultiHeadedAttention as BaseMultiHeadedAttention
from torch import nn
from torch.nn.init import xavier_uniform_


class StyleEncoder(pl.LightningModule):
    def __init__(
        self,
        idim=80,
        gst_tokens=10,
        gst_token_dim=256,
        gst_heads=4,
        conv_layers=6,
        conv_chans_list=(32, 32, 64, 64, 128, 128),
        conv_kernel_size=3,
        conv_stride=2,
        gru_layers=1,
        gru_units=128,
    ):
        super().__init__()

        self.ref_enc = ReferenceEncoder(
            idim=idim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
        )
        self.stl = StyleTokenLayer(
            ref_embed_dim=gru_units,
            gst_tokens=gst_tokens,
            gst_token_dim=gst_token_dim,
            gst_heads=gst_heads,
        )

    def forward(self, speech):
        ref_embs = self.ref_enc(speech)
        style_embs = self.stl(ref_embs)

        return style_embs


class ReferenceEncoder(pl.LightningModule):
    def __init__(
        self,
        idim=80,
        conv_layers=6,
        conv_chans_list=(32, 32, 64, 64, 128, 128),
        conv_kernel_size=3,
        conv_stride=2,
        gru_layers=1,
        gru_units=128,
    ):
        super().__init__()

        assert conv_kernel_size % 2 == 1, "the size of kernel must be odd."
        assert (
            len(conv_chans_list) == conv_layers
        ), "the number of conv layers must be the same as the length of channels list."

        self.convs = nn.Sequential()
        padding = (conv_kernel_size - 1) // 2
        for i in range(conv_layers):
            conv_in_chans = 1 if i == 0 else conv_chans_list[i - 1]
            conv_out_chans = conv_chans_list[i]
            self.convs.add_module(
                f"conv{i}",
                nn.Conv2d(
                    conv_in_chans,
                    conv_out_chans,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding,
                    bias=False,
                ),
            )
            self.convs.add_module(f"batch_norm{i}", nn.BatchNorm2d(conv_out_chans))
            self.convs.add_module(f"relu{i}", nn.ReLU(inplace=True))

        self.conv_layers = conv_layers
        self.kernel_size = conv_kernel_size
        self.stride = conv_stride
        self.padding = padding

        gru_in_units = idim
        for i in range(conv_layers):
            gru_in_units = (
                gru_in_units - conv_kernel_size + 2 * padding
            ) // conv_stride + 1
        gru_in_units *= conv_out_chans
        self.gru = nn.GRU(gru_in_units, gru_units, gru_layers, batch_first=True)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "conv" in name and "weight" in name:
                xavier_uniform_(param.data)

    def forward(self, speech):
        batch_size = speech.size(0)
        xs = speech.unsqueeze(1)
        hs = self.convs(xs).transpose(1, 2)
        time_length = hs.size(1)
        hs = hs.contiguous().view(batch_size, time_length, -1)
        self.gru.flatten_parameters()
        _, ref_embs = self.gru(hs)
        ref_embs = ref_embs[-1]

        return ref_embs


class StyleTokenLayer(pl.LightningModule):
    def __init__(
        self,
        ref_embed_dim=128,
        gst_tokens=10,
        gst_token_dim=256,
        gst_heads=4,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.ref_embed_dim = ref_embed_dim
        self.gst_tokens = gst_tokens
        self.gst_token_dim = gst_token_dim
        self.gst_heads = gst_heads
        self.dropout_rate = dropout_rate

        gst_embs = torch.randn(gst_tokens, gst_token_dim // gst_heads)
        self.gst_embs = nn.Parameter(gst_embs)
        self.mha = MultiHeadedAttention(
            n_head=gst_heads,
            n_feat=gst_token_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, ref_embs):
        batch_size = ref_embs.size(0)
        gst_embs = torch.tanh(self.gst_embs).unsqueeze(0).expand(batch_size, -1, -1)
        ref_embs = ref_embs.unsqueeze(1)
        style_embs = self.mha(ref_embs, gst_embs, gst_embs, None)

        return style_embs.squeeze(1)


class MultiHeadedAttention(pl.LightningModule, BaseMultiHeadedAttention):
    def __init__(self, q_dim, k_dim, v_dim, n_head, n_feat, dropout_rate=0.0):
        super().__init__(n_head, n_feat, dropout_rate)
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = torch.nn.Linear(q_dim, n_feat)
        self.linear_k = torch.nn.Linear(k_dim, n_feat)
        self.linear_v = torch.nn.Linear(v_dim, n_feat)
        self.linear_out = torch.nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)
