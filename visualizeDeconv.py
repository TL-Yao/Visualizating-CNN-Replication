import torch
from networks.deConvNetwork import DeconvNet
from networks.convNetwork import ConvNet
import cv2
import numpy as np
import sys
import torchvision
from preProcessing import get_transform
from matplotlib import pyplot as plt
from torchvision import utils
import torch.nn.functional as F
import torch
from scipy.stats import mode
from PIL import Image
import math
from matplotlib.gridspec import GridSpec
from networks.customAlexNet import CustomAlexNet
import argparse


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_conv_model(model_path=None):
    # model = ConvNet(hook=True)
    # model.load_state_dict(torch.load('./model.pth', map_location=get_device(), weights_only=True))
    if not model_path:
        model = CustomAlexNet()
    else:
        model = ConvNet(hook=True)
        model.load_state_dict(torch.load(model_path, map_location=get_device(), weights_only=True))

    model.eval()
    return model


def get_deconv_model(conv_model):
    model = DeconvNet(conv_model)
    model.eval()
    return model


def load_data_loader(dataset_root_path):
    transform = get_transform()

    # load validation set
    valset = torchvision.datasets.ImageNet(
        root=dataset_root_path, split="val", transform=transform
    )

    # get all labels
    labels = np.unique(valset.targets)
    np.random.shuffle(labels)  # shuffle labels

    # create a new dataset, ensuring that each batch belongs to the same label
    class_indices = {
        label: np.where(np.array(valset.targets) == label)[0] for label in labels
    }

    # create a new index list
    indices = []
    for label in labels:
        # select 50 samples for each label
        selected_indices = np.random.choice(
            class_indices[label], size=min(50, len(class_indices[label])), replace=False
        )
        indices.append(selected_indices)

    # merge all indices into a one-dimensional array
    indices = np.concatenate(indices)

    # create a new dataset
    new_valset = torch.utils.data.Subset(valset, indices)

    return torch.utils.data.DataLoader(new_valset, batch_size=50, shuffle=False)


def visualize_img_in_channel(deconv_imgs, original_imgs, crop=False, kernal=None):
    crop_coords = []
    deconv_imgs = normalize_img(deconv_imgs)
    original_imgs = normalize_img(original_imgs)
    if not crop:
        grid_deconv_imgs = utils.make_grid(
            deconv_imgs, nrow=3, normalize=False, padding=1
        )
        grid_original_imgs = utils.make_grid(
            original_imgs, nrow=3, normalize=False, padding=1
        )
    else:
        # crop deconv imgs
        cropped_deconv_imgs = []
        for deconv_img in deconv_imgs:
            cropped_deconv_img, coords = crop_deconv_img(deconv_img)
            cropped_deconv_imgs.append(torch.from_numpy(cropped_deconv_img))
            crop_coords.append(coords)

        # crop original imgs
        cropped_original_imgs = []
        for i, original_img in enumerate(original_imgs):
            cropped_original_img = original_img[
                :,
                crop_coords[i][1] : crop_coords[i][3],
                crop_coords[i][0] : crop_coords[i][2],
            ]
            cropped_original_imgs.append(cropped_original_img)

        # make all cropped deconv imgs have the same size
        max_height = max(img.shape[1] for img in cropped_deconv_imgs)
        max_width = max(img.shape[2] for img in cropped_deconv_imgs)

        resized_deconv_imgs = []
        resized_original_imgs = []
        for i, deconv_img in enumerate(cropped_deconv_imgs):
            resized_deconv_img = (
                F.interpolate(
                    deconv_img.unsqueeze(0),
                    size=(max_height, max_width),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .clamp(0, 1)
            )
            resized_deconv_img = augment_feature_map(resized_deconv_img)
            resized_deconv_imgs.append(resized_deconv_img)
            resized_original_img = (
                F.interpolate(
                    cropped_original_imgs[i].unsqueeze(0),
                    size=(max_height, max_width),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .clamp(0, 1)
            )
            resized_original_imgs.append(resized_original_img)

        # make grid
        grid_deconv_imgs = utils.make_grid(
            resized_deconv_imgs, nrow=3, normalize=False, padding=1
        )
        grid_original_imgs = utils.make_grid(
            resized_original_imgs, nrow=3, normalize=False, padding=1
        )

    grid_kernal = (
        utils.make_grid(kernal, nrow=1, normalize=True, padding=1)
        if kernal is not None and len(kernal) > 0
        else None
    )

    return grid_deconv_imgs, grid_original_imgs, grid_kernal


def visualize_img(feature_maps, selected_batch_imgs, kernals, crop=False, layer=None):
    print(len(feature_maps), len(selected_batch_imgs), len(kernals))
    deconv_img_grids = []
    original_img_grids = []
    kernal_grids = []
    padding = 1
    for feature_map, selected_batch_img, kernal in zip(
        feature_maps, selected_batch_imgs, kernals
    ):
        deconv_img_grid, original_img_grid, kernal_grid = visualize_img_in_channel(
            feature_map, selected_batch_img, crop, kernal
        )
        deconv_img_grids.append(deconv_img_grid)
        original_img_grids.append(original_img_grid)
        if kernal_grid is not None:
            kernal_grids.append(kernal_grid)

    # make all cropped deconv imgs have the same size
    max_height = max(img.shape[1] for img in deconv_img_grids)
    max_width = max(img.shape[2] for img in deconv_img_grids)

    resized_deconv_imgs = []
    resized_original_imgs = []
    for i, deconv_img in enumerate(deconv_img_grids):
        resized_tensor = F.interpolate(
            deconv_img.unsqueeze(0),
            size=(max_height, max_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        resized_deconv_imgs.append(resized_tensor)
        resized_original_img = F.interpolate(
            original_img_grids[i].unsqueeze(0),
            size=(max_height, max_width),
            mode="bilinear",
            align_corners=False,
        )
        resized_original_imgs.append(resized_original_img.squeeze(0))

    grid_deconv_imgs = utils.make_grid(
        resized_deconv_imgs,
        nrow=math.ceil(math.sqrt(len(deconv_img_grids))),
        normalize=False,
        padding=padding,
    )
    grid_original_imgs = utils.make_grid(
        resized_original_imgs,
        nrow=math.ceil(math.sqrt(len(original_img_grids))),
        normalize=False,
        padding=padding,
    )
    grid_kernal = (
        utils.make_grid(
            kernal_grids,
            nrow=math.ceil(math.sqrt(len(kernal_grids))),
            normalize=False,
            padding=padding,
        )
        if kernal_grids
        else None
    )

    size = math.ceil(math.sqrt(len(deconv_img_grids)))

    if grid_kernal is not None:
        fig = plt.figure(figsize=(12, 6))
        fig.patch.set_facecolor("black")
        plt.suptitle(f"Layer {layer}", color="white")
        gs = GridSpec(size, 2 * size)
        # Kernel grid 3x3 on left
        kernal_axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]
        # Original images 3x3 on right
        original_axes = fig.add_subplot(gs[:, 3:])

        for i in range(3):
            for j in range(3):
                kernal_axes[i][j].imshow(
                    kernal_grids[i * 3 + j].numpy().transpose((1, 2, 0))[:, :, ::-1],
                    interpolation="bilinear",
                )
                kernal_axes[i][j].axis("off")

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05)

        original_axes.imshow(
            grid_original_imgs.numpy().transpose((1, 2, 0))[:, :, ::-1]
        )
        original_axes.axis("off")
    else:
        fig, axs = plt.subplots(
            1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1]}
        )
        fig.patch.set_facecolor("black")
        plt.suptitle(f"Layer {layer}", color="white")
        axs[0].imshow(
            grid_deconv_imgs.numpy().transpose((1, 2, 0))[:, :, ::-1],
            interpolation="bilinear",
        )
        axs[0].set_title("Deconv Images")

        axs[1].imshow(
            grid_original_imgs.numpy().transpose((1, 2, 0))[:, :, ::-1],
            interpolation="bilinear",
        )
        axs[1].set_title("Original Images")

    plt.tight_layout()
    plt.show()


def augment_feature_map(feature_map_tensor):
    feature_map_np = feature_map_tensor.numpy().transpose(1, 2, 0)
    feature_map_np = feature_map_np[:, :, ::-1]  # RGB to BGR
    low_pass = cv2.GaussianBlur(feature_map_np, (5, 5), 0)
    enhanced_map = cv2.addWeighted(feature_map_np, 1.5, low_pass, -0.5, 0)
    enhanced_map = torch.from_numpy(enhanced_map.transpose(2, 0, 1))  # BGR to RGB
    return enhanced_map


def normalize_img(img):
    for c in range(img.shape[0]):
        img[c] = (img[c] - img[c].min()) / (img[c].max() - img[c].min())
    return img


def crop_deconv_img(deconv_img):
    if type(deconv_img) == torch.Tensor:
        deconv_img = deconv_img.cpu().detach().numpy().transpose(1, 2, 0)
    # calculate the mode of each channel
    mode_values = mode(
        deconv_img.reshape(-1, deconv_img.shape[-1]), axis=0, keepdims=True
    ).mode[0]

    # create a mask to mark non-background pixels
    mask = np.any(deconv_img != mode_values, axis=-1)
    coords = np.argwhere(mask)

    if coords.size > 0:
        # get the bounding box of the non-background area
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        # make the bounding box a square
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        side_length = max(width, height)

        # calculate the difference between the side length and the width or height
        width_diff = side_length - width
        height_diff = side_length - height

        # expand the bounding box on both sides
        x_min = max(x_min - width_diff // 2, 0)
        x_max = min(x_max + width_diff - width_diff // 2, deconv_img.shape[1] - 1)
        y_min = max(y_min - height_diff // 2, 0)
        y_max = min(y_max + height_diff - height_diff // 2, deconv_img.shape[0] - 1)

        # calculate the center of the square
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2

        # calculate the starting and ending coordinates of the square
        y_start = max(center_y - side_length // 2, 0)
        x_start = max(center_x - side_length // 2, 0)
        y_end = y_start + side_length
        x_end = x_start + side_length

        # make sure the coordinates do not exceed the image boundaries while still being a square
        if y_end > deconv_img.shape[0]:
            y_start = deconv_img.shape[0] - side_length
            y_end = deconv_img.shape[0]
        if x_end > deconv_img.shape[1]:
            x_start = deconv_img.shape[1] - side_length
            x_end = deconv_img.shape[1]

        # crop the image to a square
        deconv_img = deconv_img[y_start:y_end, x_start:x_end]
    else:
        # return the original image if no non-background pixels are found
        x_start, y_start, x_end, y_end = 0, 0, deconv_img.shape[1], deconv_img.shape[0]

    return deconv_img.transpose(2, 0, 1), (x_start, y_start, x_end, y_end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="path to the model")
    parser.add_argument("--dataset", type=str, default="./data", help="path to the dataset")
    args = parser.parse_args()

    np.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(threshold=sys.maxsize)
    conv_model = get_conv_model(args.model)
    deconv_model = get_deconv_model(conv_model)
    data_loader = load_data_loader(args.dataset)
    data_loader_iter = iter(data_loader)

    for layer in range(6):
        feature_maps = []
        selected_batch_imgs = []
        kernals = []
        iterat_count = 1 if layer == 0 else 4
        for _ in range(iterat_count):
            img_batch, label = next(data_loader_iter)
            result = deconv_model(img_batch, layer)
            feature_map, selected_batch_idx, kernal = (
                result["feature_map"],
                result["selected_batch_idx"],
                result["kernal"],
            )
            if layer != 0:
                feature_maps.append(feature_map[0])
                selected_batch_imgs.append(img_batch[selected_batch_idx[0]])
                kernals.append(None)
            else:
                feature_maps = feature_map
                selected_batch_imgs = [
                    img_batch[idxs] for idxs in selected_batch_idx
                ]
                kernals = kernal

        visualize_img(
            feature_maps, selected_batch_imgs, kernals, crop=True, layer=layer
        )
