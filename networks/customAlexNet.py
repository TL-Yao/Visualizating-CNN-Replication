import torch
import torch.nn as nn
import torchvision.models as models
from networks.switch import Switch
import torch.nn.functional as F


class CustomAlexNet(nn.Module):
    def __init__(self):
        super(CustomAlexNet, self).__init__()
        # load pretrained alexnet model
        alexnet = models.alexnet(pretrained=True)

        # extract feature layers
        self.layer0 = nn.Sequential(alexnet.features[0], alexnet.features[1])
        self.layer1 = nn.Sequential(
            alexnet.features[2], alexnet.features[3], alexnet.features[4]
        )
        self.layer2 = nn.Sequential(
            alexnet.features[5], alexnet.features[6], alexnet.features[7]
        )
        self.layer3 = nn.Sequential(alexnet.features[8], alexnet.features[9])
        self.layer4 = nn.Sequential(alexnet.features[10], alexnet.features[11])
        self.layer5 = nn.Sequential(alexnet.features[12])
        # extract classifier layers
        self.classifier = alexnet.classifier

        # for storing switch information of max pooling layers
        self.switches = Switch()

        self._register_hooks()

    def _register_hooks(self):
        for i, layer in enumerate(self.children()):
            for sub_layer in layer:
                if isinstance(sub_layer, nn.MaxPool2d):
                    sub_layer.register_forward_hook(self._hook_switches(i))

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
