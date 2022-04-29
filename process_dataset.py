from data_processor import DataProcessor
import numpy as np
import os

SHUFFLE = True
NORM_METHOD = "non_zero_min_max" # None, min_max, non_zero_min_max
EXPORT_SUFFIX = "processed" # will be added to old file names
DATA_FOLDER = "F:/music_datasets/dev_dataset/"
TRAIN_DATA_NAMES = ()