from dataset_creator import Dataset
from data_processor import DataProcessor, normalize
import numpy as np
import json
import os


#################
## BASIC SETUP ##
#################

# Define dataset paths
BASE_FOLDER = "F:/music_datasets/dev_dataset/"
RAW_MP3_FOLDER = "raw_mp3s/"

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

SLICED_MP3_FOLDER = "processed_mp3s/"
SLICE_DURATION = 15 # in seconds

# Adjust parameters to your liking
D.create_mp3_dataset(target_path=D.dataset_folder+SLICED_MP3_FOLDER, slice_duration=SLICE_DURATION, max_slices=2, random_slice_selection=True, overlap=0, normalize_mp3s=True)

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

SPEC_FOLDER = "spectrograms/"

D.create_melspec_dataset(target_path = D.dataset_folder+SPEC_FOLDER,
                         sr = 22050, hop_length = 1024, n_fft = 2048, n_mels = 90)

print("Spectrograms created")
print()


####################
## AGGREGATE DATA ##
####################

# Generate a dictionary for train-validation-test split
'''
Custom splits are possible if given in the form:
{category_1: {"val":[val_track_name_1, val_track_name_2, ...],
             "test":[test_track_name_1, test_track_name_2, ...]},
category_2: ...
}
val_track_name refers to the track name WITHOUT .mp3 ending.
'''

train_test_val_dict = D.create_train_val_test_dict(relative_sizes=(0.8,0.1,0.1), seed=10)
with open(D.dataset_folder+"train_test_val_dict.json", "w") as file:
    json.dump(train_test_val_dict, file)


