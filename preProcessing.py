"""
According to the paper, Images are preprocessed resizing the smallest dimension to 256, cropping the 
center 256x256 region, subtracting the per-pixel mean (across all images) and then using 10 diﬀerent 
sub-crops of size 224x224 (corners + center with(out) horizontal flips).

not sure if the author is using random crop or ten crops. [Krizhevsky et al., 2012] used random crop 
in training process and ten crops in testing process. But author says they used ten crops in training process.
I don't have enough disk space to store ten crops, so I use random crop in training process.
"""

import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import torchvision


def _random_select_crop(crops):
    # Randomly select one crop
    selected_crop = crops[torch.randint(len(crops), (1,)).item()]
    # Check if the selected crop is already a tensor
    if not isinstance(selected_crop, torch.Tensor):
        # Convert the selected crop to a tensor
        selected_crop = torchvision.transforms.functional.to_tensor(selected_crop)
    return selected_crop


def get_transform():
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop(256),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.ToTensor(),
            transforms.Lambda(lambda x: x[[2, 1, 0], ...]),
            transforms.Lambda(
                lambda x: x - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            ),  # subtract mean
        ]
    )

    return transform
