"""self implementation."""
from typing import Optional, List

import torch
import numpy as np
from PIL import Image
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
        self.mean_precisions = []
        self.mean_recalls = []

    def run(self):
        """Run a single epoch loop.

        Returns:
            Mean absolute error.
        """
        # reset current results and add next entry for running loss
        self.true_values = []
        self.predicted_values = []

        self.best_true_values = []
        self.best_predicted_values = []

        self.precision_values = []
        self.recall_values = []

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
                true = true.detach().cpu().numpy().squeeze()
                predicted = predicted.detach().cpu().numpy().squeeze()

                # generate a density map by applying a Gaussian filter
                true_gauss = gaussian_filter(true, sigma=(1, 1), order=0)

                true_counts = np.sum(true_gauss) / 100
                predicted_counts = np.sum(predicted) / 100

                # print("counts", true_counts, predicted_counts)

                # update current epoch results
                self.true_values.append(true_counts)
                self.predicted_values.append(predicted_counts)


                precision, recall = find_precision_recall(true, predicted)

                self.precision_values.append(precision)
                self.recall_values.append(recall)

        # localization error and precision
        self.update_precision()

        # calculate errors and standard deviation
        self.update_errors()

        return self.mean_abs_err

    def update_best_values(self): 
        self.best_true_values = self.true_values.copy()
        self.best_predicted_values = self.predicted_values.copy()

    def update_precision(self): 
        mean_precision = sum(self.precision_values) / self.size
        self.mean_precisions.append(mean_precision)

        mean_recall = sum(self.recall_values) / self.size
        self.mean_recalls.append(mean_recall)

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

    def get_results(self):
        return (
            f"{'Train' if not self.validation else 'Valid'}:\n"
            f"\tAverage loss: {self.running_loss[-1]:3.4f}\n"
            f"\tMean error: {self.mean_err:3.3f}\n"
            f"\tMean absolute error: {self.mean_abs_err:3.3f}\n"
            f"\tError deviation: {self.std:3.3f}\n"
            f"\tMean precision: {self.mean_precisions[-1]:3.3f}\n"
            f"\tMean recall: {self.mean_recalls[-1]:3.3f}\n"
        )

def find_precision_recall(true, predicted):
    n = int(np.sum(predicted) / 100)

    peaks = peak_local_max(predicted, exclude_border=False, num_peaks=n)
    x = peaks[:, 0]
    y = peaks[:, 1]

    dmap = np.full(predicted.shape, 0)
    dmap[x, y] = 1

    EXPANSION = 5 
    true_exp = maximum_filter(true, size=(EXPANSION,)*2)
    dmap_exp = maximum_filter(dmap, size=(EXPANSION,)*2)

    #find true positives, false positives and false negatives
    TP = np.count_nonzero(np.logical_and(true, dmap_exp)) 
    FP = np.count_nonzero(np.logical_and(true_exp == 0, dmap)) 
    FN = np.count_nonzero(np.logical_and(true, dmap_exp == 0)) 

    try:
      precision = TP / (TP + FP)
      recall = TP / (TP + FN)

    except ZeroDivisionError: 
      return 0, 0

    return precision, recall 


def test_precision_recall():
    TP = FP = FN = 2
    true = np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0]])
    predicted = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])

    precision, recall = find_precision_recall(true, predicted)

    assert precision == TP / (TP + FP)
    assert recall == TP / (TP + FN)

#1st, 11th, 18th
#find time to present article yourself