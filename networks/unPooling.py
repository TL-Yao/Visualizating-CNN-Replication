"""
Unpooling: In the convnet, the max pooling operation is non-invertible, however we can obtain an approximate 
inverse by recording the locations of the maxima within each pooling region in a set of switch variables. 
In the deconvnet, the unpooling operation uses these switches to place the reconstructions from the layer 
above into appropriate locations, preserving the structure of the stimulus. 
"""

import torch.nn as nn
import torch


class Unpooling(nn.Module):
    def __init__(self, pool_layer):
        """
        initialize unpooling operation
        :param pool_layer: original pooling layer, used to get parameters
        """
        super(Unpooling, self).__init__()
        self.kernel_size = pool_layer.kernel_size
        self.stride = pool_layer.stride
        self.padding = pool_layer.padding

    def forward(self, feature_map, switches, output_size):
        """
        execute unpooling operation
        :param feature_map: feature map
        :param switches: switch information recorded during pooling
        :param output_size: output size of the unpooling operation,
        :return: unpooled feature map
        """
        # Get shape information
        batch_size, channels, height, width = feature_map.size()
        height_out, width_out = output_size

        # initialize output tensor
        output = torch.zeros(
            (
                batch_size,
                channels,
                height_out + self.padding * 2,
                width_out + self.padding * 2,
            )
        )

        # flatten feature_map and switches for scatter operation
        flat_feature_map = feature_map.view(batch_size, channels, -1)
        flat_switches = switches.view(batch_size, channels, -1)

        # Iterate over each element in the flattened tensors
        for i in range(flat_switches.shape[-1]):
            # Calculate window starting positions
            h_start = (i // (height_out // self.stride)) * self.stride
            w_start = (i % (width_out // self.stride)) * self.stride

            # Map local indices to global positions
            h_offset = flat_switches[:, :, i] // self.kernel_size
            w_offset = flat_switches[:, :, i] % self.kernel_size

            # Compute final positions in the output tensor
            h_pos = h_start + h_offset
            w_pos = w_start + w_offset

            # Update output tensor with input values
            output[
                torch.arange(batch_size)[:, None],
                torch.arange(channels)[None, :],
                h_pos,
                w_pos,
            ] += flat_feature_map[:, :, i]

        # remove padding
        if self.padding > 0:
            output = output[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]

        return output
