# Single Label Audio Processing Pipeline (SLAPP)
Author: _Max Hilsdorf_

-----------------------------------
## 1. What is SLAPP?
__SLAPP__ is a data processing pipeline suitable for a wide range of single label audio classification tasks. Currently, it is able to transform a dataset of labeled MP3 files into a dataset of mel spectrograms split into train-, validation- and test data.

-----------------------------------

## 2. Why SLAPP?

Audio-based deep learning is a challenging field of study. One reason for that is that after acquiring a suitable mp3 dataset, there are still many steps to take until you can start training your first neural network. SLAPP handles all of the following steps for you:
1. Cut mp3s down to uniform length
2. Extract multiple audio slices from each MP3
3. Perform a train-validation-test split
4. Convert mp3 slices to mel spectrograms
5. Aggregate mel spectrograms to numpy datasets
6. Normalize and shuffle numpy datasets

-----------------------------------

## 3. Advantages of SLAPP

### Intuitive Input Format
SLAPP does not require a dataframe with file paths and labels as an input. Instead, just throw your audio files into a folder corresponding to their label. SLAPP will handle the rest for you.

### Stores Intermediate Results

SLAPP does not just provide you with your processed and ready-to-use dataset. Instead, your data will be saved along each major processing step (audio slicing, spectrogram computation, data aggregation, data processing). This way, SLAPP makes the entire process transparent and allows you to observe each individual data point at all processing stages.

### Optimized for Local Computation

SLAPP is made to use on a home pc and requires neither a GPU nor any cloud computation services. If your dataset is large and the processing takes too long, just shut down your computer and resume the data processing another time. SLAPP will reload your progress and make sure none of your precious time is lost. SLAPP was specifically built for researchers, freelancers or hobby programmers who do not have access to advanced processing ressources.

### Smart Train-Validation-Test Split
If you draw multiple snippets from an audio file, you have to ensure that all slices of an input file are located EITHER in the train-, validation-, or test dataset. If that is not the case, the validation- and test data are not sufficiently independent, which may distort your classification results. SLAPP makes sure that no snippets from the same audio file land in separate data splits.

### Customizable
All relevant pipeline parameters can be set in ```pipeline_parameters.json```. This includes:
* duration and number of audio slices to extract
* other audio slicing features like random slice selection or overlapping slices
* spectrogram creation parameters
* data normalization methods

Further customizations include:
* using a custom train-validation-test dict
* implementing other image-like audio representations like MFCCs or Chromagrams

-----------------------------------

## 4. Which Problems Can SLAPP Solve?

SLAPP can solve any problem with the following attributes:
* Input consists only of mp3 files
* Each input file has exactly ONE corresponding label

Do not use SLAPP if:
* You are not trying to predict classes, but continuous numeric values (regression problem)
* You have a multiclass problem

You should make use of SLAPPs audio slicing functionality if all or most of the possible slices of each input file share the same label. E.g., a music genre classifier or a speaker sex classifier are a great fit, while e.g. the emotional profile of a piece of music or a speech sample can vary drastically within the recording.

-----------------------------------

## 5. Modular Components

SLAPP consists of the modules: ```audio_processor.py```, ```data_processor.py```, ```dataset_creator.py```.

### Audio Processor (audio_processor.py)
Includes utility functions for all processing steps applied directly to audio files (except for audio augmentation, which has its own module).
* ```slice_mp3```

Cuts an audio file into several slices of the same length using _pydub.AudioSegment_.

* ```create_melspectrogram```

Converts an audio slice into a melspectrogram using _librosa.features.melspectrogram_.

### Data Processor (data_processor.py)
Includes a DataProcessor class built around processing seperate numpy arrays for train-, val-, and test data.
* ```DataProcessor.normalize_data```

Normalizes the data for better training convergence. Currently implemened: _min_max_ (scaling to interval [0,1]) and _non_zero_min_max_ (scaling interval to [0.001, 1.001])
* ```DataProcessor.shuffle_data```
* ```DataProcessor.export_data```

### Dataset Creator (dataset_creator.py)
This module is the heart of SLAPP. It processes a dataset from raw audio files to ready-to-use numpy arrays while maintaining control over every processing step as well as allowing to exit and reload midst processing steps. <br>
A ```Dataset``` class is instantiated from nothing but a local folder containing another folder with all your mp3 files structured by their class label.

* ```Dataset.create_mp3_dataset```

Converts a dataset of raw mp3s into a sliced-mp3 dataset of uniform audio length.

* ```Dataset.create_melspec_dataset```

Converts a dataset of sliced mp3s into a melspectrogram dataset.

* ```Dataset.create_training_dataset```

Aggregates a melspectrogram and performs a track-wise train-val-test split (meaning no two slices from the same audio file are in different splits). Saves the aggregated and split datasets as numpy arrays.

-----------------------------------

## 6. Requirements

As far as python libraries go, only three libraries outside of the Python Standard Library are needed:
* numpy
* librosa
* pydub

Additionally, the audio codec [FFmpeg](https://ffmpeg.org/download.html) must be installed on your system.

-----------------------------------

## 7. How to Use SLAPP

First, make sure all the requirements are installed (Section 6). Also download (or clone) SLAPP into a directory of your choice, which we will call "slapp_dir/". I am going to explain how to use SLAPP in formal and less-formal terms but also give an easy example.

### 7.1. Required Dataset Structure
Let's start with a formal explanation. For SLAPP to be applicable, you need to have a dataset of size $N$ with $k$ different classes/labels. Every MP3 audio file $x_1, x_2, ..., x_n$ belongs to exactly one of the classes $(c_1, ..., c_k)$. This results in a dataset $D$ where every audio file $x_i$ is associated with a class label $c_i$. This means that $D_i = (x_i, c_{x_i})$ for $i \leq N$.
In a directory of your choice, let's call it "data_dir/", open a folder "data_dir/raw_mp3s/" (other names are also possible but must be configured in ```pipeline_parameters.json```). For every class $c_i \in (c_1, ..., c_k)$ open a directory "data_dir/raw_mp3s/c_i/". Each audio file $x_i$ must be placed in the directory "data_dir/raw_mp3s/c_{x_i}/x_i". <br> <br>
In less formal terms, imagine you have lots of MP3 files which all belong to exactly one out of at least two classes. Open a new directory "data_dir/" for your dataset as well as a folder "raw_mp3s/" within this directory. Within the folder "raw_mp3s", open another folder named after each class in your dataset and put all the MP3s which belong to this class in there. <br> <br>
As an easy example, imagine you have 100 speech files, where 50 were recorded by males and 50 were recorded by females. In order to build a classifier for female vs. male speech, open a directory "data_dir/raw_mp3s/male/" and a directory "data_dir/raw_mp3s/female/". Now, throw the 50 male-spoken files in the male folder and the 50 female-spoken files in the female folder.

### 7.2. Set Pipeline Parameters
To configure the pipeline according to your needs, open "slapp_dir/pipeline_parameters.json" with the text editor of your choice. Set the first part of the JSON file like this:
```
"project_folders": {
    "base_folder": "<full path to your dataset>/",
    "raw_mp3_folder": "raw_mp3s/" ( or whatever you called your mp3 folder )
},
```
Adjust the paramaters under "build_dataset_params" to your liking:
* "sliced_mp3_folder" - _str_: folder in which to store the processed MP3s
* "slice_duration" - _int_: desired duration of all audio snippets in seconds
* "max_slices" - _int_: maximum number of snippets to draw from a track
* "overlap" - _int_: overlap between consecutive slices in full seconds
* "random_slice_selection" - _bool_: if more snippets than "max_slices" are available, draw randomly or from the start?
* "normalize_mp3s" - _bool_: bring each snippet to the maximum volume or not
* "spec_folder" - _str_: folder in which to store the mel spectrograms
* "sample_rate" - _int_: sampling rate
* "hop_length" - _int_: hop length
* "n_fft" - _int_: frame size
* "n_mels" - _int_: number of mel filter banks
* "custom_train_val_test_dict": _bool_ or _dict_: advanced feature which allows you to supply your custom train-validation-test split dict
* "data_folder": _str_ - folder to store the train-, validation- and test datasets in

Under "process_dataset_params", set the following parameters to your liking:
* "norm_method": _str_ ("non_zero_min_max" or "min_max" currently implemented): data normalization method
* "shuffle": _bool_: whether to shuffle the datasets or not
* "export_suffix": _str_: suffix to be appended to distinguish processed from unprocessed datasets

### 7.3. Run SLAPP

Run ```build_dataset.py``` and then ```process_dataset.py``` to build and process your dataset. If the dataset building takes too long, you can terminate the program at any time and restart it to resume your progress. After running both files, the processed mel spectrograms, as well as the class labels, will be stored (separately for train-, validation-, and test split) in the "data_folder" you chose in ```pipeline_parameters.json```.

-----------------------------------

## 8. Current Status
SLAPP is feature-complete concerning all aspects laid out in Sections 3 to 5. Here are some plans I have on how to refine and expand SLAPP in the future:
* Implementing an audio augmentation module to increase dataset size and build more robust models
* Ensuring compatability of SLAPP with WAV files
* Reworking SLAPP such that other features than mel spectrograms (e.g. MFCCs or chromagrams) can also be computed.

-----------------------------------

## 9. Credits
SLAPP is the result of a study project at Justus-Liebig University of Gießen (GER). I want to thank my primary supervisor Prof. Ralf Köhl and my secondary supervisor Prof. Daniel Kaiser for their support and interest in my work.

-----------------------------------

## 10. License

MIT License<br>

Copyright (c) 2022 Max Hilsdorf<br>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:<br>

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.<br>

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
