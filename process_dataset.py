from data_processor import DataProcessor
import numpy as np
import os

SHUFFLE = True
NORM_METHOD = "non_zero_min_max" # None, min_max, non_zero_min_max
EXPORT_SUFFIX = "processed" # will be added to old file names
DATA_FOLDER = "F:/music_datasets/dev_dataset/training_data/"
TRAIN_DATA_NAMES = ("specs_train.npy", "labels_train.npy")
VAL_DATA_NAMES = ("specs_val.npy", "labels_val.npy")
TEST_DATA_NAMES = ("specs_test.npy", "labels_test.npy")


# Initialize Data Processor

D = DataProcessor(data_path=DATA_FOLDER, train_data_names=TRAIN_DATA_NAMES, export_path=DATA_FOLDER,
                val_data_names=VAL_DATA_NAMES, test_data_names=TEST_DATA_NAMES,export_suffix=EXPORT_SUFFIX)

# Normalization

if NORM_METHOD:
    D.normalize_data(NORM_METHOD)

# Shuffle Data
if SHUFFLE:
    D.shuffle_data(seed=10)

# Export Data
D.export_data()
