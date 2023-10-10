from torch.nn import Conv1d, ConvTranspose1d, Sequential
from torch.nn.utils import weight_norm as wnorm


def create_depthwise_separable_convolution(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode="zeros",
    transpose=False,
    weight_norm=True,
):
    if transpose:
        depth = ConvTranspose1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            bias=bias,
            groups=in_channels,
        )
        point = ConvTranspose1d(in_channels, out_channels, kernel_size=1)
    else:
        depth = Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            bias=bias,
            groups=in_channels,
        )
        point = Conv1d(in_channels, out_channels, kernel_size=1)
    if weight_norm:
        return Sequential(wnorm(depth), wnorm(point))
    else:
        return Sequential(depth, point)
