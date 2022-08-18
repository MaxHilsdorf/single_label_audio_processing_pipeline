# Single Label Audio Processing Pipeline (SLAPP)
Author: _Max Hilsdorf_

-----------------------------------
## 1. What is SLAPP?
__SLAPP__ is a data processing pipeline suitable for a wide range of single label audio classification tasks. Currently, it is able to transform a dataset of labeled mp3 files into a dataset of melspectrograms split into train-, validation- and test data. SLAPP is a __work-in-progress__ project. Although its main components are already functional, I am still working on a clean documentation and a suitable repository structure to make SLAPP as easy to use as possible. The full release is scheduled for __6th of September 2022__. See section 7 for current status updates.

-----------------------------------

## 2. Why SLAPP?

Audio based deep learning is a challenging field of study. One reason for that is that after aquiring a suitable mp3 dataset, there are still many steps to take until you can start training your first neural network. SLAPP handels all of the following steps for you:
1. Cut mp3s down to uniform length
2. Extract multiple audio slices from each mp3
3. Perform a train-validation-test split
4. Convert mp3 slices to melspectrograms
5. Aggregate melspectrograms to numpy datasets
6. Normalize and shuffle numpy datasets
7. Augment the train dataset using a wide range of audio-effects

-----------------------------------

## 3. Advantages of SLAPP

### Intuitive Input Format
SLAPP does not require a dataframe with file paths and labels as an input. Instead, just throw your audio files into a folder corresponding to their label. SLAPP will handle the rest for you.

### Zwischenschritte [ENG]

SLAPP does not just provide you with your processed and ready-to-use dataset. Instead, your data will be saved along each major processing step (audio slicing, spectrogram computation, data aggregation, data processing). This way, SLAPP makes the entire process transparent and allows you to observe each individual data point at all processing stages.

### Optimized for Local Computation

SLAPP is made to use on a home pc and requires neither a GPU nor any cloud computation services. If your dataset is large and the processing takes too long, just shut down your computer and resume the data processing another time. SLAPP will reload your progress and make sure none of your precious time is lost. SLAPP was specifically built for researchers, freelancers or hobby programmers who do not have access to advanced processing ressources.

### Sophisticated Augmentation Features
SLAPP makes the most out of your data by:
* extracting multiple audio slices from each track with our without overlap
* augmenting your audio files by randomly applying carefully selected effect chains

To keep your models clean, SLAPP makes sure that:
* all slices of an input file are located EITHER in the train-, validation-, or test dataset
* effect-based audio augmentations are only computed for files in the train dataset 

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

SLAPP consists of the modules: ```audio_processor.py```, ```data_processor.py```, ```dataset_creator.py```, ```audio_augmenter.py```.

### Audio Processor (audio_processor.py)
Includes utility functions for all processing steps applied directly to audio files (except for audio augmentation, which has its own module).
* ```slice_mp3```

Cuts an audio file into several slices of the same length using _pydub.AudioSegment_.

* ```create_melspectrogram```

Converts an audio slice into a melspectrogram using _librosa.features.melspectrogram_.

### Data Processor (data_processor.py)
Included a DataProcessor class built around processing seperate numpy arrays for train-, val-, and test data.
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

Aggregates a melspectrogram and performs a base-file-wise train-val-test split. Saves the aggregated and split datasets as numpy arrays.


### Audio Augmenter (audio_augmenter.py)

-----------------------------------

## 6. How to Use SLAPP

-----------------------------------

## 7. Current Status
Currently, every SLAPP feature is implemented in principle. However, there are some constraints:
* Documentation of all modules and scripts is still work-in-progress.
* Currently no script for applying the ```audio_augmenter.py``` module to a full dataset.
* Exception handling must be refined. Currently, corrupt mp3 files can break the pipeline.
* Overall, the module is not yet tested on a large enough variety of use cases.
* Please remember the official release date is 06. September 2022

-----------------------------------

## 8. Credits
SLAPP is the result of a study project at Justus-Liebig University of Gießen (GER). I want to thank my supervisor Prof. Ralf Köhl for his professional advice as well as for his trust in my work. Also, I thank Prof. Daniel Kaiser for his interest in my work in his role as a secondary supervisor.

-----------------------------------

## 9. License

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