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
    help="The probability of horizontal flip for tratining dataset.",
)
@click.option(
    "-rt",
    "--rotation_chance",
    default=0.0,
    help="The chance of a 2D rotation for the training dataset."
)
@click.option(
    "--unet_filters",
    default=64,
    help="Number of filters for U-Net convolutional layers.",
)
@click.option(
    "--convolutions", default=2, help="Number of layers in a convolutional block."
)
@click.option("--verbose", is_flag=True, help="Whether to log the detailed training information to console.")
@click.option("-s", "--save", type=click.Path(exists=False), help="Save plot and log data to dataset folder.")
def train(
    data_path: str,
    network_architecture: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    horizontal_flip: float,
    vertical_flip: float,
    rotation_chance: float,
    unet_filters: int,
    convolutions: int,
    verbose: bool,
    save: str,
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
            rotation_chance if mode == "train" else 0
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

    log_file = None

    if save is not None:
        os.makedirs(save, exist_ok=True)

        id = 1
        log_path = os.path.join(save, f"log{id}.txt")
        while os.path.exists(log_path):
            id += 1
            log_path = os.path.join(save, f"log{id}.txt")

        
        log_file = open(log_path, "a")

    # current best results (lowest mean absolute error on validation set)
    best_valid_mae = np.infty

    for epoch in range(epochs):
        _log(f"Epoch {epoch + 1}", log_file, True) #always print this

        # run training epoch and update learning rate
        train_looper.run()
        lr_scheduler.step()

        # run validation epoch
        with torch.no_grad():
            best_result = valid_looper.run()

        _log(train_looper.get_results(), log_file, verbose)
        _log(valid_looper.get_results(), log_file, verbose)

        # update checkpoint if new best is reached
        if best_result < best_valid_mae:
            best_valid_mae = best_result

            train_looper.update_best_values()
            valid_looper.update_best_values()

            torch.save(
                network.state_dict(),
                os.path.join(data_path, f"{network_architecture}.pth"),
            )

            _log(f"New best result: {best_result}", log_file, verbose)

        _log("-" * 50, log_file, verbose)

    if save is not None:
        _plot(train_looper, save)
        _plot(valid_looper, save)

    _log(f"[Training done] Best valid MAE: {best_valid_mae}", log_file, True)
    _log(f"[Training done] Best train MAE: {train_looper.best_mae}", log_file, True)

    _log(f"[Training done] Best valid Precision: {valid_looper.best_precision}", log_file, True)
    _log(f"[Training done] Best valid Recall: {valid_looper.best_recall}", log_file, True)

def _log(text: str, log_file=None, verbose=False):
    """Print text to file or CLI or both, depending on the options."""
    
    if log_file is not None: print(text, file=log_file)
    if verbose: print(text)
    

def _plot(looper: Looper, path):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    """Plot true vs predicted counts and loss."""
    # true vs predicted counts
    true_line = [[0, max(looper.best_true_values)]] * 2  # y = x
    ax[0].cla()
    ax[0].set_title('Train' if not looper.validation else 'Valid')
    ax[0].set_xlabel('True value')
    ax[0].set_ylabel('Predicted value')
    ax[0].plot(*true_line, 'r-')
    ax[0].scatter(looper.best_true_values, looper.best_predicted_values)

    # loss
    epochs = np.arange(1, len(looper.running_loss) + 1)
    ax[1].cla()
    ax[1].set_title('Train' if not looper.validation else 'Valid')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].plot(epochs, looper.running_loss, "r", label="Loss")

    #precision and recall
    ax[2].set_ylabel("Evaluation (%)")
    ax[2].plot(epochs, looper.mean_precisions, "b", label="Precision")
    ax[2].plot(epochs, looper.mean_recalls, "g", label="Recall")
    ax[2].legend()

    prefix = "train" if not looper.validation else "valid"

    id = 1
    save_path = os.path.join(path, f"{prefix}{id}.png")

    while os.path.exists(save_path):
        id += 1
        save_path = os.path.join(path, f"{prefix}{id}.png")

    fig.savefig(save_path, bbox_inches='tight') 

if __name__ == "__main__":
    train()
