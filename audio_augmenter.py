import numpy as np
import librosa
import soundfile
import random
from pedalboard import Pedalboard, Chorus, Reverb, Compressor

def add_white_noise(signal: np.ndarray, sr, noise_factor: float):
    noise = np.random.normal(0, signal.std(), signal.size)
    aug_signal = signal + noise * noise_factor
    return aug_signal

def slow_time(signal, sr, stretch_rate):
    n_frames = len(signal)
    stretched_signal = librosa.effects.time_stretch(signal, stretch_rate)
    return stretched_signal[:n_frames]

def scale_pitch(signal, sr, n_semitones):
    return librosa.effects.pitch_shift(signal, sr, n_semitones)

def invert_polarity(signal, sr):
    return signal * -1

def random_gain(signal, sr, min_gain_factor, max_gain_factor):
    gain_factor = random.uniform(min_gain_factor, max_gain_factor)
    return signal * gain_factor

def compress_signal(signal, sr, threshold, ratio):
    board = Pedalboard([Compressor(threshold_db=threshold, ratio=ratio)])
    return librosa.util.normalize(board(signal, sr))

def add_chorus(signal, sr, rate_hz, depth):
    board = Pedalboard([Chorus(rate_hz=rate_hz, depth=depth)])
    return board(signal, sr)

def add_reverb(signal, sr, room_size, damping, wet_level, dry_level):
    board = Pedalboard([Reverb(room_size=room_size, damping=damping, wet_level=wet_level,
                                dry_level=dry_level)])
    return librosa.util.normalize(board(signal, sr))


class AudioAugmenter:

    def __init__(self, effect_dict={"white_noise":0.6, "slow_time":0.6, "scale_pitch":0.6, "invert_polarity":0.3,
            "random_gain":0.6, "compressor":0.6, "chorus":0.5, "reverb":0.6},
            n_max_effects = 4, n_min_effects = 1):

        self.effect_dict = effect_dict
        self.effects = list(self.effect_dict.keys())
        self.probs = list(self.effect_dict.values())
        self.n_max_effects = n_max_effects
        self.n_min_effects = n_min_effects

        self.effect_settings = {
            "white_noise":{"noise_factor":(0.025,0.075)},
            "slow_time":{"stretch_rate":(0.95,0.99)},
            "scale_pitch":{"n_semitones":(-3,3)},
            "invert_polarity":True,
            "random_gain":{"min_gain_factor":2, "max_gain_factor":4},
            "compressor":{"threshold":(-30,-10), "ratio":(2,5)},
            "chorus":{"rate_hz":(0.5,0.9), "depth":(0.1,0.2)},
            "reverb":{"room_size":(0.1,0.5), "damping":(0.3,0.9),
                    "wet_level":(0.3,0.7), "dry_level":(0.3,0.7)}
        }

        self.effect_callables = {"white_noise":add_white_noise, "slow_time":slow_time, "scale_pitch":scale_pitch,
                                "invert_polarity":invert_polarity, "random_gain":random_gain, "compressor":compress_signal,
                                "chorus":add_chorus, "reverb":add_reverb}

    def roll_effect_board(self):

        effect_board = {}

        if random.random() < self.effect_dict["white_noise"]:
            effect_board["white_noise"] = tuple((random.uniform(*self.effect_settings["white_noise"]["noise_factor"]),))
        if random.random() < self.effect_dict["slow_time"]:
            effect_board["slow_time"] = tuple((random.uniform(*self.effect_settings["slow_time"]["stretch_rate"]),))
        if random.random() < self.effect_dict["scale_pitch"]:
            effect_board["scale_pitch"] = tuple((random.randint(*self.effect_settings["scale_pitch"]["n_semitones"]),))
        if random.random() < self.effect_dict["invert_polarity"]:
            effect_board["random_gain"] = tuple(self.effect_settings["random_gain"].values())
        if random.random() < self.effect_dict["compressor"]:
            effect_board["compressor"] = (random.uniform(*self.effect_settings["compressor"]["threshold"]),
                                        random.uniform(*self.effect_settings["compressor"]["ratio"]))
        if random.random() < self.effect_dict["chorus"]:
            effect_board["chorus"] = (random.uniform(*self.effect_settings["chorus"]["rate_hz"]),
                                    random.uniform(*self.effect_settings["chorus"]["depth"]))
        if random.random() < self.effect_dict["reverb"]:
            effect_board["reverb"] = (np.random.uniform(*self.effect_settings["reverb"]["room_size"]),
                                    np.random.uniform(*self.effect_settings["reverb"]["damping"]),
                                    np.random.uniform(*self.effect_settings["reverb"]["wet_level"]),
                                    np.random.uniform(*self.effect_settings["reverb"]["dry_level"]))
        return effect_board



    def roll_effect_order(self, effect_board):

        effect_list = list(effect_board.keys())
        random.shuffle(effect_list)

        return effect_list


    def apply_random_effect_board(self, signal, sr):

        effect_board = self.roll_effect_board()
        effect_order = self.roll_effect_order(effect_board)

        for effect in effect_order:

            #print(effect, effect_board[effect])

            signal = self.effect_callables[effect](signal, sr, *effect_board[effect])

        return signal

    def export_signal(self, signal, sr, export_path):
        soundfile.write(export_path, signal, samplerate=sr)
