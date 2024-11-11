import time

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from layers.snippets import Transpose
import torch.nn.functional as F
from typing import Optional


# class Conv1d(nn.Conv1d):
#     def _conv_forward(
#         self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
#     ) -> Tensor:
#         return super()._conv_forward(
#             x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
#         )


class Conv1dSubampling_new(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduce_time_layers: int = 2) -> None:
        super(Conv1dSubampling_new, self).__init__()

        # First, reduce the time_length
        time_reduce_layers = []
        for _ in range(reduce_time_layers):
            time_reduce_layers.extend([
                nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                nn.GELU()
            ])
        self.time_reduce = nn.Sequential(*time_reduce_layers)

        # Then, mix the model_dim
        self.dim_mix = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, inputs: Tensor) -> (Tensor, Tensor):
        inputs = inputs.permute(0, 2, 1)  # Change shape to (batch_size, dim, time)
        tokens = self.time_reduce(inputs)
        outputs = self.dim_mix(tokens)
        outputs = outputs.permute(0, 2, 1)  # Revert shape to (batch_size, time, dim)
        return outputs, tokens.permute(0, 2, 1)


class Conv1dSubampling(nn.Module):
    """
    Convolutional 1d subsampling with padding to control sequence length reduction.
    Args:
        in_channels (int): Number of channels in the input (e.g., n_mels for spectrogram)
        out_channels (int): Number of channels produced by the convolution (typically model dimension)
        reduce_time_layers (int): Number of halving conv layers to apply (default is 2 for 1/4 reduction)

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs

    Returns:
        - **outputs** (batch, time, dim): Tensor produced by the convolution
    """

    def __init__(self, in_channels: int, out_channels: int, reduce_time_layers: int = 2) -> None:
        super(Conv1dSubampling, self).__init__()

        layers = [nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1), nn.GELU()]

        for _ in range(reduce_time_layers):
            layers.extend([
                nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.GELU()
            ])

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs.permute(0, 2, 1)  # Change shape to (batch_size, dim, time)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # Revert shape to (batch_size, time, dim)
        return x


class Conv1dUpsampling(nn.Module):
    def __init__(self, hidden_dim: int, reduce_time_layers: int = 2):
        super(Conv1dUpsampling, self).__init__()

        # Upsample only in the time dimension, increase time dimensions of the hidden_states tensor
        layers = []
        for _ in range(reduce_time_layers):
            layers.extend([
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GELU()
            ])
        self.time_upsample = nn.Sequential(*layers)

        # Reduce the potential effects of padded artifacts introduced by the upsampling
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, dim, time)
        x = self.time_upsample(x)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # Revert shape to (batch_size, time, dim)
        return x



# Test the PaddedConvSubampling and UpsamplingModule

# Generate random input data
batch_size = 64  # 32
seq_length = 4096 # 160
in_channels = 12
out_channels = 256
reduce_time_layers = 2

inputs = torch.rand(batch_size, seq_length, in_channels)
print(f"Input shape before processing: {inputs.shape}")

# Initialize the PaddedConvSubampling module
subsampling_layer = Conv1dSubampling_new(in_channels=in_channels, out_channels=out_channels, reduce_time_layers=2)

# Pass the input through the PaddedConvSubampling module
subsampling_output, tokens = subsampling_layer(inputs)
print(f"Output shape after subsampling: {subsampling_output.shape}")
print(tokens.shape)

# Update the input_projection layer
input_dropout_p = 0.1  # This value was taken from the original code you shared
hidden_dim = out_channels  # This is typically the model dimension (e.g., 512)
input_projection = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),  # maps from d_model to d_model
    nn.Dropout(p=input_dropout_p),
)

# Pass the output from subsampling through the input_projection
projected_output = input_projection(subsampling_output)
print(f"Output shape after input projection: {projected_output.shape}")

# Initialize the UpsamplingModule
upsampling_layer = Conv1dUpsampling(hidden_dim=hidden_dim, reduce_time_layers=2)

# Pass the output from projection through the UpsamplingModule
upsampled_output = upsampling_layer(projected_output)
print(f"Output shape after upsampling: {upsampled_output.shape}")

# project the hidden dimensions back to the original features
output_projection = nn.Linear(hidden_dim, in_channels)
outputs = output_projection(upsampled_output)
print(f"Output shape after processing: {outputs.shape}")

# Compare the shapes
assert inputs.shape == outputs.shape, "Original input and upsampled output shapes do not match!"