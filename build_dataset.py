from dataset_creator import Dataset
import numpy as np
import json
import os


#####################
## LOAD PARAMETERS ##
#####################

with open("pipeline_parameters.json", "r") as file:
    param_dict = json.load(file)


#################
## BASIC SETUP ##
#################

BASE_FOLDER = param_dict["project_folders"]["base_folder"]
RAW_MP3_FOLDER = param_dict["project_folders"]["raw_mp3_folder"]

# Initiate Dataset instance
D = Dataset(BASE_FOLDER+RAW_MP3_FOLDER)

# Check Dataset structure valid
dataset_valid = D.get_structure_report()

if not dataset_valid:
    exit()


################
## SPLIT MP3S ##
################

print()
print(D.line)
print("PROCESSING MP3s")
print(D.line)
print()

# Load params
slice_params = param_dict["build_dataset_params"]["audio_slicing_params"]

SLICED_MP3_FOLDER = slice_params["sliced_mp3_folder"]
SLICE_DURATION = slice_params["slice_duration"]
MAX_SLICES = slice_params["max_slices"]
OVERLAP = slice_params["overlap"]
RANDOM_SLICE_SELECTION = slice_params["random_slice_selection"]
NORMALIZE_MP3S = slice_params["normalize_mp3s"]

# Adjust parameters to your liking
D.create_mp3_dataset(target_path=D.dataset_folder+SLICED_MP3_FOLDER, slice_duration=SLICE_DURATION,
                    max_slices=MAX_SLICES, random_slice_selection=RANDOM_SLICE_SELECTION,
                    overlap=OVERLAP, normalize_mp3s=NORMALIZE_MP3S)

print(D.line)
print("MP3s processed")
print("Sliced MP3 Distribution:")
for cat in D.categories:
    n_mp3s = len(os.listdir(D.dataset_folder+SLICED_MP3_FOLDER+cat+"/"))
    print(f"{cat}: {n_mp3s}")
print()


############################
## CREATE MELSPECTROGRAMS ##
############################

print()
print(D.line)
print("CREATING SPECTROGRAMS")
print(D.line)
print()

# Load params
spec_params = param_dict["build_dataset_params"]["melspec_creation_params"]

SPEC_FOLDER = spec_params["spec_folder"]
SAMPLE_RATE = spec_params["sample_rate"]
HOP_LENGTH = spec_params["hop_length"]
N_FFT = spec_params["n_fft"]
N_MELS = spec_params["n_mels"]

D.create_melspec_dataset(target_path = D.dataset_folder+SPEC_FOLDER,
                         sr = SAMPLE_RATE, hop_length = HOP_LENGTH,
                         n_fft = N_FFT, n_mels = N_MELS)

print("Spectrograms created")
print()


####################
## AGGREGATE DATA ##
####################

print(D.line)
print("SPEC AGGREGATION")
print(D.line)
print()

# Load params
agg_params = param_dict["build_dataset_params"]["melspec_aggregation_params"]

CUSTOM_DICT = agg_params["custom_train_val_test_dict"]
DATA_FOLDER = agg_params["data_folder"]

# Generate a dictionary for train-validation-test split
'''
Custom splits are possible if given in the form:
{category_1: {"val":[val_track_name_1, val_track_name_2, ...],
             "test":[test_track_name_1, test_track_name_2, ...]},
category_2: ...
}
val_track_name refers to the track name WITHOUT .mp3 ending.
'''

if not CUSTOM_DICT:
    train_val_test_dict = D.create_train_val_test_dict(relative_sizes=(0.8,0.1,0.1), seed=10)
with open(D.dataset_folder+"train_val_test_dict.json", "w") as file:
    json.dump(train_val_test_dict, file)

D.create_training_datasets(D.dataset_folder+DATA_FOLDER, train_val_test_dict, assert_shape=None, bit=16)

print("Specs aggregated")
print()

