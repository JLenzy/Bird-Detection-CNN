# Freefield1010 dataset
# https://arxiv.org/abs/1309.5275

import os
import torch.cuda
from torch.utils.data import Dataset
import torchaudio
import pandas as pd

class FreeField1010Dataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_directory,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):

        self.annotations = pd.read_csv(annotations_file)
        self.audio_directory = audio_directory
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        # return number of samples in dataset
        return len(self.annotations)

    def __getitem__(self, index):
        # __get__item is used to retrieve index item (list[0])
        # freefield[0] will return first index item

        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        label = float(label)
        signal, sr = torchaudio.load(audio_sample_path) #tensor [channels, samples]
        signal = signal.to(self.device)

        # resample
        if sr != self.target_sample_rate:
            signal = self._resample(signal, sr)
        # convert to mono
        if signal.shape[0] > 1:
            signal = self._mixdown(signal)
        # reduce length to desired # of samples
        if signal.shape[1] > self.num_samples:
            signal = self._cutlength(signal)
        # 0-pad shorter samples
        if signal.shape[1] < self.num_samples:
            signal = self._right_padding(signal)

        # convert to MelSpectrogram
        signal = self.transformation(signal)

        return signal, label

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_directory,
                            str(self.annotations.iloc[index, 0]))
        path = path + ".wav"
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]

    def _resample(self, signal, sr):
        resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
        return resampler(signal)

    def _mixdown(self, signal):
        return torch.mean(signal, dim=0, keepdim=True)

    def _cutlength(self, signal):
        return signal[:, :self.num_samples]

    def _right_padding(self, signal):
        length_signal = signal.shape[1]
        num_missing_samples = self.num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    # convert to melspectrogram and return


if __name__ == "__main__":

    ANNOTATIONS_FILE = "/Users/jlenz/Desktop/Datasets/BirdAudioDetection/metadata.csv"
    AUDIO_DIRECTORY ="/Users/jlenz/Desktop/Datasets/BirdAudioDetection/wav"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 44100

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device {device}.")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64)

    ffbirds = FreeField1010Dataset(ANNOTATIONS_FILE,
                                   AUDIO_DIRECTORY,
                                   mel_spectrogram,
                                   SAMPLE_RATE,
                                   NUM_SAMPLES,
                                   device)

    print(f"There are {len(ffbirds)} samples in this dataset.")

    index = 6
    mel_spec, label = ffbirds[index]

    print(f"Signal {index} has label {label}")
    print(f"Number of Channels: {mel_spec.shape[0]}")
    print(f"Shape of signal: {mel_spec.shape}")
    print(f"Length of File: {mel_spec.shape[1]} Samples ({mel_spec.shape[1] / SAMPLE_RATE} seconds)")

