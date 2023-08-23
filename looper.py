"""self implementation."""
from typing import Optional, List

import torch
import numpy as np
import os
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage.feature import peak_local_max


class Looper:
    """self handles epoch loops, logging, and plotting."""

    def __init__(
        self,
        network: torch.nn.Module,
        device: torch.device,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: torch.utils.data.DataLoader,
        dataset_size: int,
        validation: bool = False,
    ):
        """
        Initialize self.

        Args:
            network: already initialized model
            device: a device model is working on
            loss: the cost function
            optimizer: already initialized optimizer link to network parameters
            data_loader: already initialized data loader
            dataset_size: no. of samples in dataset
            plot: matplotlib axes
            validation: flag to set train or eval mode

        """
        self.network = network
        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.loader = data_loader
        self.size = dataset_size
        self.validation = validation
        self.running_loss = []

    def run(self):
        """Run a single epoch loop.

        Returns:
            Mean absolute error.
        """
        # reset current results and add next entry for running loss
        self.true_values = []
        self.predicted_values = []

        self.precisions = []
        self.recalls = []

        self.running_loss.append(0)

        # set a proper mode: train or eval
        self.network.train(not self.validation)

        for image, label in self.loader:
            # move images and labels to given device
            image = image.to(self.device)
            label = label.to(self.device)

            # clear accumulated gradient if in train mode
            if not self.validation:
                self.optimizer.zero_grad()

            # get model prediction (a density map)
            result = self.network(image)

            # calculate loss and update running loss
            loss = self.loss(result, label)
            self.running_loss[-1] += image.shape[0] * loss.item() / self.size

            # update weights if in train mode
            if not self.validation:
                loss.backward()
                self.optimizer.step()

            # loop over batch samples
            for true, predicted in zip(label, result):
                # integrate a density map to get no. of objects
                # note: density maps were normalized to 100 * no. of objects
                #       to make network learn better

                # localization error and precision
                self.update_precision(true, predicted)

                # generate a density map by applying a Gaussian filter
                predicted = gaussian_filter(predicted, sigma=(1, 1), order=0)

                true_counts = torch.sum(true).item() / 100
                predicted_counts = torch.sum(predicted).item() / 100

                # update current epoch results
                self.true_values.append(true_counts)
                self.predicted_values.append(predicted_counts)

        # calculate errors and standard deviation
        self.update_errors()

        return self.mean_abs_err

    def update_precision(self, true, predicted):
        true = true.detach().cpu().numpy()
        predicted = predicted.detach().cpu().numpy()

        precision, recall = find_precision_recall(true, predicted)

        self.precisions.append(precision)
        self.recalls.append(recall)

    def update_errors(self):
        """
        Calculate errors and standard deviation based on current
        true and predicted values.
        """
        self.err = [
            true - predicted
            for true, predicted in zip(self.true_values, self.predicted_values)
        ]
        self.abs_err = [abs(error) for error in self.err]
        self.mean_err = sum(self.err) / self.size
        self.mean_abs_err = sum(self.abs_err) / self.size
        self.std = np.array(self.err).std()

        # self.precision = [predicted / true for true, predicted in zip(self.true_values, self.predicted_values)]
        # self.mean_precision = sum(self.avg_precision) / self.size

        # self.recall = [predicted / true for true, predicted in zip(self.true_values, self.predicted_values)]

    def get_results(self):
        return (
            f"{'Train' if not self.validation else 'Valid'}:\n"
            f"\tAverage loss: {self.running_loss[-1]:3.4f}\n"
            f"\tMean error: {self.mean_err:3.3f}\n"
            f"\tMean absolute error: {self.mean_abs_err:3.3f}\n"
            f"\tError deviation: {self.std:3.3f}\n"
        )


def find_precision_recall(true, predicted):
    # make a charitable dmap of detected objects
    UNCERTAINTY = 3
    dmap = maximum_filter(predicted, size=(UNCERTAINTY,) * 2)
    print(dmap)

    THRESHOLD = 10

    #find true positives, false positives and false negatives
    TP = np.count_nonzero(np.logical_and(true, dmap >= THRESHOLD))
    FP = np.count_nonzero(np.logical_and(true, dmap < THRESHOLD))
    FN = np.count_nonzero(np.logical_and(true == 0, dmap >= THRESHOLD))

    print(f"TP: {TP}, FP: {FP}, FN: {FN}")

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return precision, recall 


def test_precision_recall():
    TP = FP = FN = 2
    true = np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0]])
    predicted = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])

    precision, recall = find_precision_recall(true, predicted)

    assert precision == TP / (TP + FP)
    assert recall == TP / (TP + FN)
