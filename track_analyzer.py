
from pydub import AudioSegment
from data_processor import normalize
import librosa
import numpy as np
import audioread
import os


class Track:

    def __init__(self, mp3_path: str, assert_shape: tuple, sr: int=22050, name: str=None, slice_duration: int=10, overlap: int=0,
                    hop_length=1024, n_fft=2048, n_mels=100):

        self.mp3_path = mp3_path
        self.sr = sr
        if name:
            self.name = name
        else:
            self.name = ".".join(self.mp3_path.split("/")[-1].split(".")[:-1])
        self.spec_shape = assert_shape
        self.audio_segment = AudioSegment.from_mp3(self.mp3_path).set_frame_rate(int(sr/2))
        #self.audio_segment = AudioSegment.from_mp3(self.mp3_path)
        self.duration = int(self.audio_segment.duration_seconds) # round down
        self.sub_segments = self.build_segments(slice_duration=slice_duration, overlap=overlap,
                            hop_length=hop_length, n_fft=n_fft, n_mels=n_mels, assert_shape=self.spec_shape)


    def build_segments(self, slice_duration: int, overlap: int=0,
                        hop_length=1024, n_fft=2048, n_mels=100, assert_shape=None):

        # Compute timestamps
        slice_timestamps =[(0, slice_duration*1000)] # add first sample
        for i in range(slice_duration-overlap,self.duration-overlap,slice_duration-overlap): # add the rest
            slice_timestamps.append((i*1000, ((i+slice_duration)*1000)))

        # Build Segments
        sub_segments = []
        for ts in slice_timestamps:
            sub_segments.append(TrackSegment(self, timestamp=ts, hop_length=hop_length,
                                                n_fft=n_fft, n_mels=n_mels, assert_shape=assert_shape))
        return sub_segments

class TrackSegment:

    def __init__(self, track: AudioSegment, timestamp: tuple, hop_length: int=1024, n_fft: int=2048,
                n_mels = 100, assert_shape=None):

                self.timestamp = timestamp
                self.audio_segment = track.audio_segment[timestamp[0]:timestamp[1]]
                self.waveform = np.array(self.audio_segment.get_array_of_samples(), dtype="float16")
                self.spectrogram = librosa.feature.melspectrogram(y=self.waveform, n_fft=n_fft, hop_length=hop_length,
                                                                    sr=track.sr, n_mels=n_mels)
                self.spectrogram = librosa.power_to_db(self.spectrogram, ref=np.max)

                if assert_shape and (self.spectrogram.shape[1] < assert_shape[1]):
                    self.spectrogram = np.pad(self.spectrogram, ((0,0),(0,assert_shape[1]-self.spectrogram.shape[1])))

class TrackAnalyzer:

    def __init__(self, model, track, normalize_method: str="non_zero_min_max",
                    expand_dims: bool=True, swap_axes: bool=False):

        self.model = model
        self.track = track
        self.spectrograms = self.process_spectrograms(normalize_method, expand_dims, swap_axes)

    def process_spectrograms(self,normalize_method: str="non_zero_min_max", expand_dims: bool=True, swap_axes: bool=False):

        # Aggregate spectrograms
        specs = np.zeros((len(self.track.sub_segments), self.track.spec_shape[0], self.track.spec_shape[1]))
        for i, subseg in enumerate(self.track.sub_segments):
            specs[i,:,:] = subseg.spectrogram

        # Normalize spectrograms
        if normalize_method == "min_max":
            specs = normalize(specs, "min_max")
        elif normalize_method == "non_zero_min_max":
            specs = normalize(specs, "non_zero_min_max")
        else:
            print("given normalization method does not exist. Skipping normalization")
            pass

        # Expand dims
        if expand_dims:
            specs = np.expand_dims(specs, 3)
            
        # Switch axes
        if swap_axes:
            specs = np.swapaxes(specs, 1,2)

        return specs

    def predict_sub_segments(self):

        return self.model.predict(self.spectrograms)
