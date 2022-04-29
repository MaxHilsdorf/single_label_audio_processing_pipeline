"""
Created on Thu Feb 10 17:24:00 2022

@author: Max

This module processes aggregated spectrogram datasets for training with
deep neural networks.

"""

import numpy as np

class DataProcessor:

    def __init__(self, data_path:str, train_data_names:tuple, export_path="",
    val_data_names:tuple=None, test_data_names:tuple=None, efficient_load:bool=True):

        self.data_is_split = False
        if val_data_names and test_data_names:
            self.data_is_split = True

        self.data_path=data_path
        self.export_path = export_path

        # Load data

        if self.data_is_split: # -> data is already split

            if efficient_load:
                self.train_inputs = np.load(self.data_path+train_data_names[0], mmap_mode="r+")
                self.train_targets = np.load(self.data_path+train_data_names[1], mmap_mode="r+")
                self.val_inputs = np.load(self.data_path+val_data_names[0], mmap_mode="r+")
                self.val_targets = np.load(self.data_path+val_data_names[1], mmap_mode="r+")
                self.test_inputs = np.load(self.data_path+test_data_names[0], mmap_mode="r+")
                self.test_targets = np.load(self.data_path+test_data_names[1], mmap_mode="r+")
            else:
                self.train_inputs = np.load(self.data_path+train_data_names[0])
                self.train_targets = np.load(self.data_path+train_data_names[1])
                self.val_inputs = np.load(self.data_path+val_data_names[0])
                self.val_targets = np.load(self.data_path+val_data_names[1])
                self.test_inputs = np.load(self.data_path+test_data_names[0])
                self.test_targets = np.load(self.data_path+test_data_names[1])
        else:
            if efficient_load:
                self.train_inputs = np.load(self.data_path+train_data_names[0], mmap_mode="r+")
                self.train_targets = np.load(self.data_path+train_data_names[1], mmap_mode="r+")
            else:
                self.train_inputs = np.load(self.data_path+train_data_names[0])
                self.train_targets = np.load(self.data_path+train_data_names[1])

    def normalize_data(self, method:str):
        """
        Arguments
        ---------
        <method>: Normalization method to use.
            Implemented: "min_max", "non_zero_min_max"

        Returns
        -------
        None
        """

        self.train_inputs = normalize(self.train_inputs, method=method)

        if self.data_is_split:
            self.val_inputs = normalize(self.val_inputs, method=method)
            self.test_inputs = normalize(self.test_inputs, method=method)

    def shuffle_data(self, seed):
        """
        Shuffles all available pairs of input and target data.
        """

        self.train_inputs, self.train_targets = shuffle(self.train_inputs, self.train_targets, seed=seed)

        if self.data_is_split:
            self.val_inputs, self.val_targets = shuffle(self.val_inputs, self.val_targets, seed=seed)
            self.test_inputs, self.test_targets = shuffle(self.test_inputs, self.test_targets, seed=seed)


    def export_data(self, train_data_names : tuple, val_data_names : tuple=None,
                    test_data_names : tuple=None):
        """
        Exports the processed data.
        """

        np.save(self.export_path+train_data_names[0], self.train_inputs)
        np.save(self.export_path+train_data_names[1], self.train_targets)

        if self.data_is_split:

            np.save(self.export_path+val_data_names[0], self.val_inputs)
            np.save(self.export_path+val_data_names[1], self.val_targets)

            np.save(self.export_path+test_data_names[0], self.test_inputs)
            np.save(self.export_path+test_data_names[1], self.test_targets)

######################
## HELPER FUNCTIONS ##
######################

def normalize(data: np.ndarray, method: str):
    """

    Normalizes a given array by a chosen method.

    Arguments
    ---------
    <method>: Normalization method to use.
        Implemented: "min_max", "non_zero_min_max"

    Returns
    -------
    np.ndarray
    """

    if method == "min_max":
        data_normalized = (data-data.min())/(data.max()-data.min())

    elif method == "non_zero_min_max":
        small_value = 0.001
        data_normalized = (data-data.min())/(data.max()-data.min()) + small_value

    return data_normalized


def shuffle(inputs: np.ndarray, targets: np.ndarray, seed=10):
    """

    Shuffles a pair of input and target data by the same order.

    Arguments
    ---------
    <inputs>: array of input data.
    <target>: array of target data.

    Returns
    -------
    (np.ndarray, np.ndarray)
    """

    np.random.seed(seed)
    p = np.random.permutation(inputs.shape[0])

    inputs_shuffled = inputs[p]
    targets_shuffled = targets[p]

    return inputs_shuffled, targets_shuffled
