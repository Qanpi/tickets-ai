"""This script apply a chosen model on a given image.

One needs to choose a network architecture and provide the corresponding
state dictionary.

Example:

    $ python infer.py -n UNet -c mall_UNet.pth -i seq_000001.jpg

The script also allows to visualize the results by drawing a resulting
density map on the input image.

Example:

    $ $ python infer.py -n UNet -c mall_UNet.pth -i seq_000001.jpg --visualize

"""
import os

import click
import torch
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from PIL import Image
from skimage.feature import peak_local_max

from models import UNet, FCRN_A


@click.command()
@click.option(
    "-i",
    "--infer_path",
    type=click.File("r"),
    required=True,
    help="A path to an input image to infer.",
)
@click.option(
    "-n",
    "--network_architecture",
    type=click.Choice(["UNet", "FCRN_A"]),
    required=True,
    help="Model architecture.",
)
@click.option(
    "-c",
    "--checkpoint",
    type=click.File("r"),
    required=True,
    help="A path to a checkpoint with weights.",
)
@click.option(
    "--unet_filters",
    default=64,
    help="Number of filters for U-Net convolutional layers.",
)
@click.option(
    "--convolutions", default=2, help="Number of layers in a convolutional block."
)
@click.option(
    "--one_channel",
    is_flag=True,
    help="Turn this on for one channel images (required for ucsd).",
)
@click.option(
    "--pad", is_flag=True, help="Turn on padding for input image (required for ucsd)."
)
@click.option(
    "-v",
    "--valid_path",
    type=click.File("r"),
    help="A path to an answer image containing true keypoints.",
)
@click.option("--visualize", is_flag=True, help="Visualize predicted density map.")
@click.option("--save", type=click.Path(exists=False), help="Save visualized plots to path.")

def infer(
    infer_path: str,
    valid_path: str,
    network_architecture: str,
    checkpoint: str,
    unet_filters: int,
    convolutions: int,
    one_channel: bool,
    pad: bool,
    visualize: bool,
    save: str,
):
    """Run inference for a single image."""
    # use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # only UCSD dataset provides greyscale images instead of RGB
    input_channels = 1 if one_channel else 3

    # initialize a model based on chosen network_architecture
    network = {"UNet": UNet, "FCRN_A": FCRN_A}[network_architecture](
        input_filters=input_channels, filters=unet_filters, N=convolutions
    ).to(device)

    # load provided state dictionary
    # note: by default train.py saves the model in data parallel mode
    network = torch.nn.DataParallel(network)
    network.load_state_dict(torch.load(checkpoint.name, map_location=device))
    network.eval()

    img = Image.open(infer_path.name)

    # padding was applied for ucsd images to allow down and upsampling
    if pad:
        img = Image.fromarray(np.pad(img, 1, "constant", constant_values=0))

    # network's output represents a density map
    density_map = network(TF.to_tensor(img).unsqueeze_(0))

    # note: density maps were normalized to 100 * no. of objects
    n_objects = torch.sum(density_map).item() / 100

    print(f"The number of objects found: {n_objects}")

    keypoints = None

    if valid_path is not None:
        answer_key = np.array(Image.open(valid_path.name).convert("1"))
        keypoints = np.argwhere(answer_key == 1)

        print(f"The true number of objects: {keypoints.shape[0]}")

    if visualize:
        _visualize(img, density_map.squeeze().cpu().detach().numpy(), n_objects, keypoints, save)

def _visualize(img, dmap, n, keypoints=None, save=False):
    """Draw a density map onto the image."""
    # keep the same aspect ratio as an input image
    fig, axes = plt.subplots(1, 2)

    # turn off axis ticks
    [ax.axis("off") for ax in axes]

    # display raw density map
    axes[0].imshow(dmap, cmap="hot")

    # overlay true keypoints
    if keypoints is not None:
        true_x = [xy[1] for xy in keypoints]
        true_y = [xy[0] for xy in keypoints]

        axes[0].scatter(x=true_x, y=true_y, c="none", s=50, marker="s", edgecolors="#0000ff")
        axes[1].scatter(x=true_x, y=true_y, c="none", s=50, marker="s", edgecolors="#0000ff")

    # find n_objects peaks 
    kernel = np.full((4, 4), 1)
    peaks = peak_local_max(
        dmap, footprint=kernel, min_distance=2, num_peaks=n, exclude_border=False
    )

    pred_x = [xy[1] for xy in peaks]
    pred_y = [xy[0] for xy in peaks]

    # plot density map over og image
    axes[1].imshow(img)
    axes[1].scatter(x=pred_x, y=pred_y, c="#ff0000", s=20, marker="x") 

    if save is not None:
        fig.savefig(save)

    plt.show()

if __name__ == "__main__":
    infer()
