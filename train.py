"""
Train the convolutional network. According to the paper, the network was trained on ImageNet2012 
which has over 1.3 million images and over 1000 classes. 

Stochastic gradient descent with a mini-batch size of 128 was used to update the parameters, starting 
with a learning rate of 10^{−2}, in conjunction with a momentum term of 0.9.

In paper, the author stated that they manually adjust the learning rate during training. I think it's because
Adam optimizer was released in 2014 and the paper was published in 2013. So I will use Adam optimizer here.

All weights are initialized to 10^{−2} and biases are set to 0. Renormalize each filter in the convolutional 
layers whose RMS value exceeds a fixed radius of 10^{−1} to this fixed radius. 
"""

import torch
import torch.nn as nn
import torchvision
from networks.convNetwork import ConvNet
from preProcessing import get_transform
import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import argparse
RANDOM_SEED = 1


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(device, lr=0.01, momentum=0.9):
    model = ConvNet()

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(momentum, 0.999))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    return model, criterion, optimizer


def load_dataset(dataset_root_path):
    transform = get_transform()

    # load ImageNet training set
    trainset = torchvision.datasets.ImageNet(
        root=dataset_root_path, split="train", transform=transform
    )

    # load validation set
    valset = torchvision.datasets.ImageNet(
        root=dataset_root_path, split="val", transform=transform
    )

    return trainset, valset


def collate_fn(batch):
    """
    since the image was cropped into 10 images, the shape becomes [batch_size, 10, 3, 224, 224],
    we need to flatten them into [batch_size * 10, 3, 224, 224]
    """
    images = []
    labels = []
    for imgs, label in batch:
        images.extend(imgs)
        labels.extend([label] * len(imgs))
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels


def get_data_loader(dataset, batch_size=128, shuffle=True, num_workers=4):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # collate_fn=collate_fn
    )


def renormalize_filters(model):
    fixed_radius = 1e-2  # Fixed radius
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            with torch.no_grad():
                # Get the weight tensor
                weight = (
                    layer.weight.data
                )  # Shape: [out_channels, in_channels, kernel_height, kernel_width]

                # Calculate the RMS value of each filter
                rms = torch.sqrt(
                    torch.mean(weight**2, dim=(1, 2, 3), keepdim=True)
                )  # Shape: [out_channels, 1, 1, 1]

                # Find filters that exceed the fixed radius
                exceed_mask = rms > fixed_radius  # Shape: [out_channels, 1, 1, 1]

                # Renormalize filters that exceed the fixed radius
                if exceed_mask.any():
                    print(
                        f"number of filters exceeding the threshold: {exceed_mask.sum().item()}"
                    )
                    scaling_factors = (
                        fixed_radius / rms
                    )  # Shape: [out_channels, 1, 1, 1]
                    scaling_factors = torch.where(
                        exceed_mask, scaling_factors, torch.ones_like(scaling_factors)
                    )  # Ensure no change for non-exceeding filters

                    # Apply scaling to the weight
                    layer.weight.data *= scaling_factors


def export_model(model):
    torch.save(model.state_dict(), "./model.pth")


def train(
    dataset_root_path,
    writer,
    batch_size=128,
    lr=0.01,
    momentum=0.9,
    num_epochs=10,
    num_workers=4,
):
    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading training set and validation set...")
    trainset, valset = load_dataset(dataset_root_path)

    print(f"Loading training loader and validation loader...")
    train_loader = get_data_loader(
        trainset, batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = get_data_loader(
        valset, batch_size, shuffle=False, num_workers=num_workers
    )

    print(f"Initializing model...")
    model, criterion, optimizer = init_model(device, lr, momentum)

    print(f"Training...")
    for epoch in tqdm.tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss = 0
        for i, (images, labels) in tqdm.tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc="Training on training set",
        ):
            # move data to device
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            # calculate loss
            loss = criterion(outputs, labels)

            # backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss, epoch * len(train_loader) + i)
            total_loss += loss.item()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    writer.add_scalar(
                        f"Gradient/layer{name}",
                        grad_norm,
                        epoch * len(train_loader) + i,
                    )

        renormalize_filters(model)

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train_avg", avg_loss, epoch)
        if epoch % 5 == 0:
            export_model(model)

        # evaluate on validation set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in tqdm.tqdm(
                val_loader, total=len(val_loader), desc="Evaluating on validation set"
            ):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(
                f"Accuracy of the model on the validation set: {100 * correct / total}%"
            )
            writer.add_scalar("Accuracy/val", 100 * correct / total, epoch)

    export_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="path to the dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=0.00001, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--num_epochs", type=int, default=70, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    log_dir = f'./logs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    train(
        dataset_root_path=args.dataset,
        writer=writer,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
    )
    writer.close()
    print("Done!")
