"""
the deconvnet uses transposed versions of the same filters, but applied to the rectified maps, not
the output of the layer beneath. In practice this means flipping each filter vertically and horizontally.
"""

import torch.nn as nn
import torch


class ReconstructFilter(nn.Module):
    def __init__(self, conv_layer):
        """
        initialize deconvolution operation
        :param conv_layer: original convolutional layer, used to get parameters
        """
        super(ReconstructFilter, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=conv_layer.out_channels,
            out_channels=conv_layer.in_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=False,
        )
        self.init_weights(conv_layer)

    def init_weights(self, conv_layer):
        # Transpose the weights to match the ConvTranspose2d requirements
        with torch.no_grad():
            self.deconv.weight.data = conv_layer.weight.data.clone()

    def forward(self, feature_map):
        """
        execute deconvolution operation
        :param feature_map: feature map
        :return: deconvolved feature map
        """
        return self.deconv(feature_map)
