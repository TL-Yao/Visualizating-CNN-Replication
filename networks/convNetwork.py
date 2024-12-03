"""
According to the paper Figure 3, construct the convolutional network the authors used.
Input: 224 x 224 x 3 image
Layer 0: 
    - 7 x 7 filter, stride 2, 96 output channels, padding 1(feature maps, 110 x 110 x 96) Paper did not mention padding, I add it here to align the shape 110 x 110 x 96
    - ReLU activation

Layer 1:
    - 3 x 3 max pooling with stride 2, output size 55 x 55 x 96, padding 1. Paper did not mention padding, I add it here to align the shape 55 x 55 x 96
    - contrast normalization
    - 5 x 5 filter, stride 2, 256 output channels(feature maps, 26 x 26 x 256)
    - ReLU activation
Layer 2:
    - 3 x 3 max pooling with stride 2, output size 13 x 13 x 256, padding 1. Paper did not mention padding, I add it here to align the shape 13 x 13 x 256
    - contrast normalization
    - 3 x 3 filter, stride 1, 384 output channels(feature maps, 13 x 13 x 384)
    - ReLU activation
Layer 3:
    - 3 x 3 filter, stride 1, 384 output channels(feature maps, 13 x 13 x 384)
    - ReLU activation
Layer 4:
    - 3 x 3 filter, stride 1, 256 output channels(feature maps, 13 x 13 x 256)
    - ReLU activation
    - 3 x 3 max pooling with stride 2, output size 6 x 6 x 256
Layer 5:
    - Fully connected layer, flatten the output of the previous layer to 9216 x 1
    - 4096 output units
    - ReLU activation
    - Dropout with p=0.5
Layer 6:
    - Fully connected layer, 4096 input units
    - 4096 output units
    - ReLU activation
    - Dropout with p=0.5
Layer 7:
    - Fully connected layer, 4096 input units
    - 1000 output units
    - Softmax activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.switch import Switch


class ConvNet(nn.Module):
    def __init__(self, hook=False):
        super(ConvNet, self).__init__()
        # Layer 0
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
        )
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(5),
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.ReLU(),
        )
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(5),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        # Layer 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # Layer 5
        self.layer5 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2))
        # Layer 6
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(), nn.Dropout(p=0.5)
        )
        # Layer 7
        self.fc2 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5))
        # Layer 7
        self.fc3 = nn.Sequential(
            nn.Linear(4096, 1000),
        )

        # initialize weights and biases
        self._initialize_weights()

        # used to store intermediate activations and pooling switches
        self.activations = {}
        self.switches = Switch()
        if hook:
            self._register_hooks()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _register_hooks(self):
        """
        register forward hook function for each layer
        """
        for i, layer in enumerate(self.children()):
            for sub_layer in layer:
                if isinstance(sub_layer, nn.MaxPool2d):
                    sub_layer.register_forward_hook(self._hook_switches(i))
                elif isinstance(sub_layer, (nn.Conv2d, nn.ReLU)):
                    sub_layer.register_forward_hook(self._hook_activations(i))

    def _hook_activations(self, layer_index):
        """
        create hook function for saving activation of layer
        """

        def hook(module, input, output):
            self.activations[layer_index] = output

        return hook

    def _hook_switches(self, layer_index):
        """
        create hook function for saving switch information of pooling layer
        """

        def hook(module, input, output):
            kernel_size = module.kernel_size
            padding = module.padding
            stride = module.stride
            batch_size, channels, _, _ = input[0].shape

            # Padding input
            input_padded = F.pad(input[0], (padding, padding, padding, padding))

            # Extract patches
            patches = F.unfold(input_padded, kernel_size=kernel_size, stride=stride)

            # Reshape patches for max operation
            patches = patches.view(batch_size, channels, kernel_size * kernel_size, -1)
            _, max_indices = patches.max(dim=2)  # Max values and their indices
            self.switches.add_switch(layer_index, max_indices)

        return hook

    def _hook_get_shape(self, layer_index):
        def hook(module, input, output):
            print(
                f"Layer {layer_index} input shape: {input[0].shape} output shape: {output.shape}"
            )

        return hook

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
