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
from looper import calculate_classifications, find_predicted_dots


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


    answer_key = None
    if valid_path is not None:
        answer_key = np.array(Image.open(valid_path.name))[:,:,0]

        print(f"The true number of objects: {np.count_nonzero(answer_key)}")

    if visualize:
        _visualize(img, density_map.squeeze().cpu().detach().numpy(), n_objects, answer_key, save)

def _visualize(img, dmap, n_objects, key, save=None):
    """Draw a density map onto the image."""
    # keep the same aspect ratio as an input image
    fig, axes = plt.subplots(1, 2)

    # turn off axis ticks
    [ax.axis("off") for ax in axes]

    # display raw density map
    axes[0].imshow(dmap, cmap="hot")
    # display og image 
    axes[1].imshow(img)

    dots = find_predicted_dots(dmap)

    # overlay true keypoints
    if key is not None:
        true = np.nonzero(key)

        edgeColor = "#00ff00"
        axes[0].scatter(x=true[1], y=true[0], c="none", s=50, marker="s", edgecolors=edgeColor)

        TP, FP, FN = calculate_classifications(key, dots)

        tps = np.nonzero(TP)
        fps = np.nonzero(FP)
        fns = np.nonzero(FN)

        axes[1].scatter(x=tps[1], y=tps[0], c="#00ff00", s=20, marker="x")
        axes[1].scatter(x=fps[1], y=fps[0], c="#ff0000", s=20, marker="x")
        axes[1].scatter(x=fns[1], y=fns[0], c="#ff00ff", s=20, marker="x")

    else:
        pred = np.nonzero(dots)
        axes[1].scatter(x=pred[1], y=pred[0], c="#ff0000", s=20, marker="x") 


    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    infer()
