"""Main script used to train networks."""
import os
from typing import Union, Optional, List

import click
import torch
import numpy as np
import matplotlib.pyplot as plt

from data_loader import H5Dataset
from looper import Looper
from models import UNet, FCRN_A

@click.command()
@click.argument("data_path", type=click.Path(exists=True), required=True,
)
@click.option(
    "-n",
    "--network_architecture",
    type=click.Choice(["UNet", "FCRN_A"]),
    required=True,
    help="Model to train.",
)
@click.option(
    "-lr",
    "--learning_rate",
    default=1e-2,
    help="Initial learning rate (lr_scheduler is applied).",
)
@click.option("-e", "--epochs", default=150, help="Number of training epochs.")
@click.option(
    "--batch_size",
    default=8,
    help="Batch size for both training and validation dataloaders.",
)
@click.option(
    "-hf",
    "--horizontal_flip",
    default=0.0,
    help="The probability of horizontal flip for training dataset.",
)
@click.option(
    "-vf",
    "--vertical_flip",
    default=0.0,
    help="The probability of horizontal flip for validation dataset.",
)
@click.option(
    "--unet_filters",
    default=64,
    help="Number of filters for U-Net convolutional layers.",
)
@click.option(
    "--convolutions", default=2, help="Number of layers in a convolutional block."
)
@click.option("--plot", is_flag=True, help="Generate a live plot.")
def train(
    data_path: str,
    network_architecture: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    horizontal_flip: float,
    vertical_flip: float,
    unet_filters: int,
    convolutions: int,
    plot: bool,
):
    """Train chosen model on selected dataset."""
    # use GPU if avilable
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = {}  # training and validation HDF5-based datasets
    dataloader = {}  # training and validation dataloaders

    for mode in ["train", "valid"]:
        # expected HDF5 files in dataset_name/(train | valid).h5
        path = os.path.join(data_path, f"{mode}.h5")
        # turn on flips only for training dataset
        dataset[mode] = H5Dataset(
            path,
            horizontal_flip if mode == "train" else 0,
            vertical_flip if mode == "train" else 0,
        )
        dataloader[mode] = torch.utils.data.DataLoader(
            dataset[mode], batch_size=batch_size
        )

    # only UCSD dataset provides greyscale images instead of RGB
    # input_channels = 1 if dataset_name == "ucsd" else 3
    input_channels = 3

    # initialize a model based on chosen network_architecture
    network = {"UNet": UNet, "FCRN_A": FCRN_A}[network_architecture](
        input_filters=input_channels, filters=unet_filters, N=convolutions
    ).to(device)
    network = torch.nn.DataParallel(network)

    # initialize loss, optimized and learning rate scheduler
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(
        network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


    # create training and validation Loopers to handle a single epoch
    train_looper = Looper(
        network,
        device,
        loss,
        optimizer,
        dataloader["train"],
        len(dataset["train"]),
    )
    valid_looper = Looper(
        network,
        device,
        loss,
        optimizer,
        dataloader["valid"],
        len(dataset["valid"]),
        validation=True,
    )

    # current best results (lowest mean absolute error on validation set)
    current_best = np.infty

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n")

        # run training epoch and update learning rate
        train_looper.run()
        lr_scheduler.step()

        # run validation epoch
        with torch.no_grad():
            result = valid_looper.run()

        # update checkpoint if new best is reached
        if result < current_best:
            current_best = result
            torch.save(
                network.state_dict(),
                os.path.join(data_path, f"{network_architecture}.pth"),
            )

            print(f"\nNew best result: {result}")

        print("\n", "-" * 80, "\n", sep="")

    if plot:
        _plot(train_looper)
        _plot(valid_looper)
        plt.show()

    print(f"[Training done] Best result: {current_best}")

def _plot(looper: Looper):
    fig, plots = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    """Plot true vs predicted counts and loss."""
    # true vs predicted counts
    true_line = [[0, max(looper.true_values)]] * 2  # y = x
    plots[0].cla()
    plots[0].set_title('Train' if not looper.validation else 'Valid')
    plots[0].set_xlabel('True value')
    plots[0].set_ylabel('Predicted value')
    plots[0].plot(*true_line, 'r-')
    plots[0].scatter(looper.true_values, looper.predicted_values)

    # loss
    epochs = np.arange(1, len(looper.running_loss) + 1)
    plots[1].cla()
    plots[1].set_title('Train' if not looper.validation else 'Valid')
    plots[1].set_xlabel('Epoch')
    plots[1].set_ylabel('Loss')
    plots[1].plot(epochs, looper.running_loss)

    fig.savefig("train.png" if not looper.validation else "valid.png")

if __name__ == "__main__":
    train()
