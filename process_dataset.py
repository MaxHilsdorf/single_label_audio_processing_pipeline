from data_processor import DataProcessor
import numpy as np
import json
import os

# Load Param Dict
with open("pipeline_parameters.json", "r") as file:
    param_dict = json.load(file)

# Fetch Params
BASE_FOLDER = param_dict["project_folders"]["base_folder"]
TRAIN_DATA_FOLDER = param_dict["build_dataset_params"]["melspec_aggregation_params"]["data_folder"]

SHUFFLE = param_dict["process_dataset_params"]["shuffle"]
NORM_METHOD = param_dict["process_dataset_params"]["norm_method"]
EXPORT_SUFFIX = param_dict["process_dataset_params"]["export_suffix"]
DATA_FOLDER = BASE_FOLDER+TRAIN_DATA_FOLDER
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
