# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:06:31 2021

@author: Max

This module allows the user to create datasets suited for single label music classification tasks.
If all relevant MP3-files are stored ina directory dir with the subfolders subdir_1, ... subdir_k
corresponding to the k classes for the classification task.

"""

import os
import numpy as np
from pydub import AudioSegment
import random
import audio_processor
import librosa
from audio_augmenter import AudioAugmenter
from data_processor import DataProcessor

class Dataset:

    """
    Allows the user to transform a structured raw MP3 dataset into a sliced MP3 dataset
    as well as into a melspectrogram dataset ready for classification tasks.
    """

    def __init__(self, raw_mp3_folder):
        self.raw_mp3_folder = raw_mp3_folder
        self.dataset_folder = "/".join(raw_mp3_folder.split("/")[:-2])+"/"
        self.categories = os.listdir(self.raw_mp3_folder)

        self.line = "-- "*10 # Standard line for print separation

    def get_structure_report(self):
        """
        Prints a report describing the dataset structure. Points the user to
        certain or potential problems. Takes no arguments. Returns True if structure
        is valid, else returns False.
        """

        valid = True

        print(self.line)
        print("DATASET STRUCTURE REPORT")
        print(self.line)
        print()

        # Do given folders exist?
        dataset_folder_exists = os.path.exists(self.dataset_folder)
        raw_mp3_folder_exists = os.path.exists(self.raw_mp3_folder)

        if not all((dataset_folder_exists, raw_mp3_folder_exists)):
            valid = False
            print("Not all given folders exist:")
            print("Dataset Folder:", dataset_folder_exists)
            print("Raw MP3 Folder:", raw_mp3_folder_exists)

        else:
            print("All given folders exist")
        
        print(self.line)
        
        # Are raw mp3s structured correctly in category folders

        categories = os.listdir(self.raw_mp3_folder)
        is_folder = [os.path.isdir(self.raw_mp3_folder+cat) for cat in categories]
        
        if not all(is_folder):
            valid = False
            print("There may be only folders within the raw mp3 folder")
            for b,cat in zip(is_folder, categories):
                if not b: print(cat)
            print("are not folders")

        else:
            print("Category structure is okay")

        print(self.line)
        
        if valid:
            # Print structure report
            print(f"The dataset has {len(categories)} classes")
            print("MP3 Distribution:")
            for cat in categories:
                n_mp3s = len(os.listdir(self.raw_mp3_folder+cat+'/'))
                print(f"{cat}: {n_mp3s} mp3s")

        else:
            print("Structure is not valid. Please fix reported issues.")

        print(self.line)
        
        return valid


    def create_mp3_dataset(self, target_path: str, slice_duration: int, max_slices: int,
                        random_slice_selection: bool, overlap: int, normalize_mp3s = False):

        """
        Creates a structured dataset where all input MP3s are sliced into equally
        long snippets.

        Arguments
        ---------
        <target_path>: str
            Specify a folder to create the new dataset in.
        <slice_duration>: int
            Duration of audio slices in seconds.
        <max_slices>: int or None
            Maximum amount of slices to extract. Enter None for no upper bound.
        <random_slice_selection>: bool
            Slices to extract are chosen at random from set of all possible slices if True.
        <overlap>: int
            Size of allowed overlap between track slices (in seconds).
        <normalize_mp3s>:
            (optional) Normalizes the track volume to max(volume) - 0.5.
            Default: No normalization.

        Returns
        -------
        None.
        """

        # Create all required folders
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        for cat in self.categories:
            if not os.path.exists(target_path+cat+"/"):
                os.makedirs(target_path+cat+"/")

        # For each category, slice all audio tracks
        for cat in self.categories:

            print(self.line)
            print(f"Processing {cat}")
            print()

            # Get all audio tracks in the category
            cat_tracks = os.listdir(self.raw_mp3_folder+cat)

            # Case: entire category is already processed
            # that is the case if the last track or the second to last track in the category
            #  can be found in the target path
            if("{}_1.mp3".format(cat_tracks[-1].split(".")[0]) in os.listdir(target_path+cat)
            or "{}_1.mp3".format(cat_tracks[-2].split(".")[0]) in os.listdir(target_path+cat)):
                print(cat, "fully processed")
                print("continuing with next category")
                print()
                continue

            # For each track, slice the track and export slices to target_path
            for i, track in enumerate(cat_tracks):

                # Show progess
                if i % 100 == 0:
                    print(f"processing {i}/{len(cat_tracks)}")

                """
                Subcase: track already fully processed
                -> continue with next track
                For efficiency purposes, we assume that a track with at least 3 processed slices is fully processed.
                """
                if f"{track[:-4]}_{3}.mp3" in os.listdir(
                        target_path+cat+"/"):
                    continue

                # Subcase: track not processed yet
                try:
                    audio_processor.slice_mp3(file_path = self.raw_mp3_folder+cat+"/"+track,
                                                     slice_duration = slice_duration,
                                                     max_slices = max_slices,
                                                     export_folder = target_path+"/"+cat+"/",
                                                     export_name = None,
                                                     normalize = normalize_mp3s,
                                                     random_slice_selection = random_slice_selection,
                                                     overlap = overlap)
                except audio_processor.AudioLoadError:
                    print("Unable to load", track)
                continue


            print(f"Finished processing {cat}")
            print()


    def create_melspec_dataset(self, mp3_path, target_path, sr = 22050, hop_length = 1024, n_fft = 2048,
                               n_mels = 60, assert_shape = None, bit = 16):

        """
        Creates a structured dataset where all input MP3s are transformed to melspectrograms

        Arguments
        ---------
        <mp3_path>:
            Specify where the relevant MP3 files are.
        <target_path>:
            Specify a folder to create the new dataset in.
        <sr>
            (optional) Sampling Rate.
            Default: 22050 Hz
        <hop_length>
            (optional) Hop length.
            Default: 512
        <n_fft>
            (optional) Length of FFT windows.
            Default: 2048
        <n_mels>
            (optional) Number of mel-bands.
            Default: 50
        <assert_shape>
            (optional) Asserts that the melspectrogram must be of a given shape.
            If it is not, an error is raised.
            Default: None
        Returns
        -------
        None.
        """

        # Create all required folders
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        for cat in self.categories:
            if not os.path.exists(target_path+cat+"/"):
                os.makedirs(target_path+cat+"/")

        # For each category, get melspectrograms for each track
        for cat in self.categories:

            print(f"Processing {cat}")
            print()

            # Get all audio tracks in the category
            cat_tracks = os.listdir(mp3_path+cat)

            # Case: entire category is already processed
            # that is the case if the last track or the second to last track in the category
            # can be found in the target path
            if("{}_1_melspec.npy".format("_".join(cat_tracks[-1].split(".")[0].split("_")[:-1])) in os.listdir(target_path+cat)
            or "{}_1_melspec.npy".format("_".join(cat_tracks[-2].split(".")[0].split("_")[:-1])) in os.listdir(target_path+cat)):
                print(cat, "fully processed")
                print("continuing with next category")
                print()
                continue

            # For each track, slice the track and export slices to target_path
            for i, track in enumerate(cat_tracks):

                # Show progess
                if i % 100 == 0:
                    print(f"processing {i}/{len(cat_tracks)}")

                # Case: melspec already compute
                # If so, go to next
                if f"{track[:-4]}_melspec.npy" in os.listdir(
                            target_path+cat+"/"):
                        continue

                # Get melspectrogram
                try:
                    melspec = audio_processor.create_melspectrogram(file_path = mp3_path+cat+"/"+track,
                                                   sr = sr, hop_length = hop_length,
                                                   n_fft = n_fft, n_mels = n_mels,
                                                   assert_shape = assert_shape, bit = bit)

                except audio_processor.AudioLoadError:
                        print("Unable to load", track)
                        continue

                # Store melspectrogram as array
                np.save(f"{target_path}{cat}/{track[:-4]}_melspec.npy", melspec)


    def create_training_dataset(self, melspec_path, target_path, assert_shape = None, bit = 16):
        """
        Transforms the structured melspectrogram dataset into two numpy arrays:
            1. concatenation of all melspectrograms
            2. array of category labels

        Arguments
        ---------
        <melspec_path>:
            Specify where the relevant melspectrograms are.
        <target_path>:
            Specify a folder to create the new dataset in.#
        <assert_shape>:
            Melspecs which differ from the given shape are ignored.
            Default: False
        Returns
        -------
        None.
        """

        # Write a text file assigning the categories to numbers
        with open(target_path+"category_labels.txt", "w") as file:
            for i, cat in enumerate(self.categories):
                file.write(f"{cat}, {i}\n")

        # Process category by category for shorter running time

        # For every category, consider all tracks
        for i, cat in enumerate(self.categories):


            # If category already processed, skip to next
            if f"melspecs_{cat}.npy" in os.listdir(target_path):
                print(f"{cat} fully processed. Skipping.")
                print()
                continue

            print(f"Processing {cat}")
            print()

            # Get all audio tracks in the category
            cat_tracks = os.listdir(melspec_path+cat)

            # For each track, slice the track and export slices to target_path
            for j, spec in enumerate(cat_tracks):

                if j% 250 == 0:
                    print(f"processing {j}/{len(cat_tracks)}")

                # Load melspec
                try:
                    ms = np.load(f"{melspec_path}{cat}/{spec}", allow_pickle = True).astype(f"float{bit}")
                except:
                    print(f"Could not load {spec}")
                    continue

                # If no shape assertion is given, derive shape from first melspec
                if not assert_shape and j == 0:
                    melspecs = np.empty((1, ms.shape[0], ms.shape[1]), dtype = f"float{bit}")
                    #labels = np.array([i], dtype = "int16")

                elif assert_shape and j == 0:
                    melspecs = np.empty((1, assert_shape[0], assert_shape[1]), dtype = f"float{bit}")
                    #labels = np.array([i], dtype = "int16")

                # Concatenate existing melspecs

                try: # If a melspecs shape is incorrect, skip it
                    ms = ms.reshape((1,melspecs.shape[1], melspecs.shape[2]))
                except ValueError:
                    print(f"{spec} has an incorrect shape.")
                    continue

                melspecs = np.concatenate([melspecs, ms], axis = 0)
                #labels = np.append(labels, i)


            # Store arrays
            np.save(target_path+f"melspecs_{cat}.npy", melspecs)
            #np.save(target_path+f"categories_{cat}.npy", labels_onehot)

            print(f"{cat} processed")
            print(f"spec shape was: {melspecs.shape}")
            print()


    def create_training_dataset_by_val_test_dict(self, melspec_path, target_path, val_test_dict, assert_shape = None, bit=16):
        """
        Alternative version of "create_training_dataset". This method creates seperate datasets for
        training, validation, and testing.

        Arguments
        ---------
        <melspec_path>:
            Specify where the relevant melspectrograms are.
        <target_path>:
            Specify a folder to create the new dataset in.
        <val_test_dict>:
            Dictionary of structure:
            {"validation":{"cat1":[mp3_names], "cat2":[mp3_names], ...},
            "test":{"cat1":[mp3_names], "cat2":[mp3_names], ...}}
        <assert_shape>:
            Melspecs which differ from the given shape are ignored.
            Default: False
        Returns
        -------
        None.
        """

        # Write a text file assigning the categories to numbers
        with open(target_path+"category_labels.txt", "w") as file:
            for i, cat in enumerate(self.categories):
                file.write(f"{cat}, {i}\n")

        # Process category by category for shorter running time

        # For every category, consider all tracks
        for i, cat in enumerate(self.categories):

            n_train = 0
            n_val = 0
            n_test = 0

            # If category already processed, skip to next
            if f"melspecs_train_{cat}.npy" in os.listdir(target_path):
                print(f"{cat} fully processed. Skipping.")
                print()
                continue

            print(f"Processing {cat}")
            print()

            # Get all audio tracks in the category
            cat_tracks = os.listdir(melspec_path+cat)

            # For each track, slice the track and export slices to target_path
            for j, spec in enumerate(cat_tracks):

                if j % 250 == 0:
                    print(f"processing {j}/{len(cat_tracks)}")

                # Load melspec
                try:
                    ms = np.load(f"{melspec_path}{cat}/{spec}", allow_pickle = True).astype(f"float{bit}")
                except:
                    print(f"Could not load {spec}")
                    continue

                if np.isnan(ms).sum() > 0:
                    print("NAN", spec)

                # Check if track is a validation/test track
                is_test_track = False
                is_val_track = False
                mp3_name = "_".join(spec.split("_")[:-2])+".mp3"
                #print(mp3_name)
                if mp3_name in val_test_dict["test"][cat]:
                    is_test_track = True
                    n_test += 1
                elif mp3_name in val_test_dict["validation"][cat]:
                    is_val_track = True
                    n_val += 1
                else:
                    n_train += 1
                    pass

                # If no shape assertion is given, derive shape from first melspec
                if not assert_shape and j == 0:
                    melspecs_train = np.zeros((1, ms.shape[0], ms.shape[1]), dtype=f"float{bit}")
                    melspecs_val = np.zeros((1, ms.shape[0], ms.shape[1]), dtype=f"float{bit}")
                    melspecs_test = np.zeros((1, ms.shape[0], ms.shape[1]), dtype=f"float{bit}")

                elif assert_shape and j == 0:
                    melspecs_train = np.zeros((1, assert_shape[0], assert_shape[1]), dtype=f"float{bit}")
                    melspecs_val = np.zeros((1, assert_shape[0], assert_shape[1]), dtype=f"float{bit}")
                    melspecs_test = np.zeros((1, assert_shape[0], assert_shape[1]), dtype=f"float{bit}")

                # Concatenate existing melspecs

                try: # If a melspecs shape is incorrect, skip it
                    ms = ms.reshape((1,melspecs_train.shape[1], melspecs_train.shape[2]))
                except ValueError:
                    print(f"{spec} has an incorrect shape.")
                    continue

                if is_test_track:
                    melspecs_test = np.concatenate([melspecs_test, ms], axis = 0)
                elif is_val_track:
                    melspecs_val = np.concatenate([melspecs_val, ms], axis = 0)
                else:
                    melspecs_train = np.concatenate([melspecs_train, ms], axis = 0)

            print()
            print("processed", cat)
            print("- "*10)
            print("Train Size:", n_train)
            print("Validation Size:", n_val)
            print("Test Size:", n_test)
            print("- "*10)
            print()

            # Store arrays
            np.save(target_path+f"melspecs_train_{cat}.npy", melspecs_train)
            np.save(target_path+f"melspecs_val_{cat}.npy", melspecs_val)
            np.save(target_path+f"melspecs_test_{cat}.npy", melspecs_test)
            #np.save(target_path+f"categories_train_{cat}.npy", labels_train_onehot)
            #np.save(target_path+f"categories_test_{cat}.npy", labels_test_onehot)

            print(f"{cat} processed")
            #print(f"spec shape was: {melspecs_train_normalized.shape}")
            print()



    def create_audio_augmentations(self, val_test_dict: dict, p_aug: float, n_aug: int, assert_shape: tuple,
                                    reg_mp3_folder: str="processed_mp3s/", aug_mp3_folder: str="augmented_mp3s/",
                                    aug_spec_folder: str="augmented_spectrograms/",
                                    training_data_folder: str="training_data/", bit: int = 16,
                                    sr: int=22050, hop_length: int=1024, n_fft: int=2048, n_mels: int=100):

        # Initialize AudioAugmenter
        A = AudioAugmenter()

        # Loop through all categories
        for i, cat in enumerate(self.categories):
            print(cat)

            # Get tracks to augment
            cat_tracks = os.listdir(self.dataset_folder+reg_mp3_folder+cat+"/")

            to_aug_tracks = []
            for track in cat_tracks:

                full_track_name = "_".join(track.split("_")[:-1])+".mp3"

                if full_track_name not in val_test_dict["validation"][cat] and full_track_name not in val_test_dict["test"][cat]:
                    if random.random() < p_aug:
                        to_aug_tracks.append(track)

            # Apply augmentation

            for to_aug in to_aug_tracks:

                try:
                    signal, sr = librosa.load(self.dataset_folder+reg_mp3_folder+cat+"/"+to_aug,
                                              sr = sr)
                except audio_processor.AudioLoadError:
                    print("Unable to load", to_aug)
                    continue

                for j in range(n_aug):

                    signal_aug = A.apply_random_effect_board(signal, sr)

                    try:
                        A.export_signal(signal_aug, sr, "tempfile.wav")
                    except RuntimeError:
                        print("Unable to write wav for", to_aug)
                        continue
                try:
                    AuSeg = AudioSegment.from_wav("tempfile.wav").export(
                                f"{self.dataset_folder+aug_mp3_folder}{cat}/{to_aug[:-4]}_aug_{j+1}.mp3",
                                format="mp3")
                except:
                    print("Unable to load wav for", to_aug)

        # Compute Spectrograms

        self.create_melspec_dataset(mp3_path=self.dataset_folder+aug_mp3_folder,
                                    target_path=self.dataset_folder+aug_spec_folder,
                                    sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels,
                                    assert_shape = assert_shape)

        # Aggregate Data

        melspecs = None
        labels = None

        for i, cat in enumerate(self.categories):

            cat_specs = os.listdir(self.dataset_folder+aug_spec_folder+cat+"/")

            for spec in cat_specs:

                a = np.expand_dims(np.load(self.dataset_folder+aug_spec_folder+cat+"/"+spec),0)

                if type(melspecs) == type(None):
                    melspecs = a
                else:
                    melspecs = np.concatenate([melspecs, a], axis=0)

                onehot = np.zeros(len(self.categories), dtype=f"int{bit}")
                onehot[i] = 1
                onehot = np.expand_dims(onehot,0)

                if type(labels) == type(None):
                    labels = onehot
                else:
                    labels = np.concatenate([labels, onehot], axis=0)

        np.save(self.dataset_folder+training_data_folder+"aug_melspecs_train.npy", melspecs)
        np.save(self.dataset_folder+training_data_folder+"aug_categories_train.npy", labels)

        # Process Data

        P = DataProcessor(data_path=self.dataset_folder+"training_data/",
                          train_data_names=("aug_melspecs_train.npy","aug_categories_train.npy"),
                          export_path=self.dataset_folder+"training_data/",
                          efficient_load=True
                          )

        print("... normalizing data ...")
        P.normalize_data(method="non_zero_min_max")

        print("... shuffling data ...")
        P.shuffle_data(seed=10)

        print("... exporting data ...")
        P.export_data(train_data_names=("aug_melspecs_train_processed.npy","aug_categories_train_processed.npy"))
