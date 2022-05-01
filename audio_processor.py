# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 22:24:01 2021

@author: Max

This module covers everything related to processing audio files,
e.g. audio slicing, audio quality adjustment, and creating spectrograms.

"""

from pydub import AudioSegment, effects
import librosa, librosa.display
import numpy as np
import audioread
from mutagen.id3 import ID3
from PIL import Image
import os
from io import BytesIO
import random

############################
##### HELPER FUNCTIONS #####
############################

def export(track, export_folder = "", export_name = "track.mp3", normalize = False):
    """
    Exports a pydub AudioSegment with standard or custom file path.

    Arguments
    ---------
    <track>:
        pydub AudioSegment.
    <export_folder>:
        (optional) Specify a folder export the file to.
        Default: Exports to this modules folder.
    <export_name>:
        (optional) Specify a new name for the file (without (!) file format e.g. '.mp3').
        Default: Keep the old file name.
    <normalize>:
        (optional) Normalizes the track volume to max(volume) - 0.5.
        Default: No normalization.

    Returns
    -------
    None.
    """

    # Normalize if requested
    if normalize:
        track = effects.normalize(track)

    # Pydub can only export to existing file_paths
    # If export_folder is non-existent, it must be created

    if export_folder != "":
        try:
            os.makedirs(export_folder)
        except FileExistsError:
            pass

        track.export(f"{export_folder}/{export_name}.mp3")

    else:
        track.export(f"{export_name}.mp3")


###########################
##### AUDIO SPLITTERS #####
###########################

def slice_mp3(file_path: str, slice_duration: int = 30, max_slices: int = None, export_folder: str = "",
               export_name: str = None, normalize: bool = False, random_slice_selection: bool = False,
               overlap: int = 0):
    """
    Splits a track into a maximum amount of slices of a given duration.

    Arguments
    ---------
    <file_path>: str
        Path to the file including the file name and its format. Only mp3!
    <slice_duration>: int
        (optional) Duration of each slice in seconds.
        Default: 30 seconds.
    <max_slices>: int
        (optional) Maximum number of slices to extract.
        Default: No limit.
    <export_folder>: str
        (optional) Specify a folder export the file to.
        Default: Exports to this modules folder.
    <export_name>: str
        (optional) Specify a new name for the file (without (!) file format e.g. '.mp3').
        Default: Keep the old file name.
    <normalize>: bool
        (optional) Normalizes the track volume to max(volume) - 0.5.
        Default: No normalization.
    <random_slice_selection>: bool
        (optional) If more than <max_slices> slices are extracted, determines whether the first
        <max_slices> slices or a random selection of <max_slices> slices should be chosen.
        Default: False (no random selection)
    <overlap>: int
        (optional) If non-zero positive, slices will overlap by <overlap> seconds.

    Returns
    -------
    None.
    """

    # Load track
    track = AudioSegment.from_mp3(file_path)

    # Get duration
    track_duration = int(track.duration_seconds) # round down

    # Calculate all possible full slices of 'slice_duration' from 0 secs till end
    step_size = slice_duration-overlap
    
    # Get starting points first, then add end points
    starting_points = []
    for p in range(0, track_duration, step_size):
        if p+slice_duration <= track_duration:
            starting_points.append(p*1000)

    slices = [(p,p+slice_duration*1000) for p in starting_points]

    # Apply max slice threshold if needed
    if max_slices and len(slices) > max_slices:

        # Apply random slice selection if requested
        if random_slice_selection:
            random.shuffle(slices)

        # Extract slices
        for i, (start, end) in enumerate(slices[:max_slices]):

            track_slice = track[start:end] # get slice

            if export_name:
                export(track_slice, export_folder, f"{export_name}_{i+1}",
                       normalize = normalize) # export with custom file name
            else:
                file_name = file_path.split("/")[-1][:-4]
                export(track_slice, export_folder, f"{file_name}_{i+1}",
                       normalize = normalize) # export with standard file name

    # If no max slice threshold is given or the track is short enough
    else:
        # Apply each slice to track and export
        for i, (start, end) in enumerate(slices):

            track_slice = track[start:end] # get slice

            if export_name:
                export(track_slice, export_folder, f"{export_name}_{i+1}",
                       normalize = normalize) # export with custom file name
            else:
                file_name = file_path.split("/")[-1][:-4]
                export(track_slice, export_folder, f"{file_name}_{i+1}",
                       normalize = normalize) # export with standard file name

##############################
##### FEATURE EXTRACTION #####
##############################

def create_melspectrogram(file_path, sr = 22050, hop_length = 512, n_fft = 2048,
                          n_mels = 50, plot = False, assert_shape = None, bit=16):
    """
    Extracts a spectrogram from an MP3 input track with the option to visualize it.

    Arguments
    ---------
    <file_path>
        Path to the file including the file name and its format. Only mp3!
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
    <plot>
        (optional) Plots the spectrogram using matplotlib.
        Default: False
    <assert_shape>
        (optional) Asserts that the melspectrogram must be of a given shape.
                   If it is not, an error is raised.
        Default: None

    Returns
    -------
    melspectrogram as numpy array.
    """

    # Load track into librosa (with exception handling)
    track = librosa.load(file_path, sr = sr)

    # Compute melspectrogram
    S = librosa.feature.melspectrogram(y=track[0], n_fft = n_fft, hop_length = hop_length,
                                       sr=sr, n_mels = n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Raise error if melspec shape does not satisfy shape assertion
    if assert_shape:
        assert S.shape[1] == assert_shape

    # Plot spectrogram is requested
    if plot:
        librosa.display.specshow(S_dB)

    return S_dB.astype(f"float{bit}")


############################
##### IMAGE PROCESSING #####
############################

def get_image(file_path, export_path = "cover_art.jpg"):
    """
    Extracts the album cover art from an MP3 file if possible.

    Arguments
    ---------
    <file_path>:
        Path to the file including the file name and its format. Only mp3!
    <export_path>:
        (optional) Specify a path export the image to. Include type format like '.jpg'.
        Default: Exports to this modules folder as JPG.

    Returns
    -------
    None.
    """
    # Load track information
    track = ID3(file_path)

    # Convert image info to actual image (if information available)
    try:
        pict = track.get("APIC:").data

    except AttributeError:
        print("The file does not contain an image.")
        return None

    im = Image.open(BytesIO(pict))

    # Export image
    im.save(export_path)
