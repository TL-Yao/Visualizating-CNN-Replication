"""
Rectification: The convnet uses relu non-linearities, which rectify the feature maps thus ensuring 
the feature maps are always positive. To obtain valid feature reconstructions at each layer (which 
also should be positive), we pass the reconstructed signal through a relu non-linearity.
"""

import torch.nn as nn


class Rectify(nn.Module):
    def __init__(self):
        """
        initialize ReLU correction operation
        """
        super(Rectify, self).__init__()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, feature_map):
        """
        execute correction operation, remove negative values
        :param feature_map: feature map
        :return: corrected feature map
        """
        return self.relu(feature_map)
