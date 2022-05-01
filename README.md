# Single Label Audio Processing Pipeline (SLAPP)
Author: _Max Hilsdorf_

## 1. What is SLAPP?
__SLAPP__ is a data processing pipeline suitable for a wide range of single label audio classification tasks. Currently, it is able to transform a dataset of labeled mp3 files into a dataset of melspectrograms split into train-, validation- and test data. SLAPP is a __work-in-progress__ project. Although its main components are already functional, I am still working on a clean documentation and a suitable repository structure to make SLAPP as easy to use as possible. The full release is scheduled for __6th of September 2022__. See section 7 for current status updates.

## 2. Why SLAPP?

Audio based deep learning is a challenging field of study. One reason for that is that after aquiring a suitable mp3 dataset, there are still many steps to take until you can start training your first neural network. SLAPP handels all of the following steps for you:
1. Cut mp3s down to uniform length
2. Extract multiple audio slices from each mp3
3. Perform a train-validation-test split
4. Convert mp3 slices to melspectrograms
5. Aggregate melspectrograms to numpy datasets
6. Normalize and shuffle numpy datasets
7. Augment the train dataset using a wide range of audio-effects

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

## 4. Which Problems Can SLAPP Solve?

SLAPP can solve any problem with the following attributes:
* Input consists only of mp3 files
* Each input file has exactly ONE corresponding label

Do not use SLAPP if:
* You are not trying to predict classes, but continuous numeric values (regression problem)
* You have a multiclass problem

You should make use of SLAPPs audio slicing functionality if all or most of the possible slices of each input file share the same label. E.g., a music genre classifier or a speaker sex classifier are a great fit, while e.g. the emotional profile of a piece of music or a speech sample can vary drastically within the recording.


## 5. Modular Components

SLAPP consists of the modules: ```audio_processor.py```, ```data_processor.py```, ```dataset_creator.py```, ```audio_augmenter.py```.

### Audio Processor (audio_processor.py)


### Data Processor (data_processor.py)

### Dataset Creator (dataset_creator.py)

### Audio Augmenter (audio_augmenter.py)

## 6. How to Use SLAPP

## 7. Current Status

## 8. Credits
SLAPP is the result of a study project at Justus-Liebig University of Gießen (GER). I want to thank my supervisor Prof. Ralf Köhl for his professional advice as well as for his trust in my work. Also, I thank Prof. Daniel Kaiser for his interest in my work secondary supervisor.

## 9. License
