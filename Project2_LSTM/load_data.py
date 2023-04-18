from typing import Tuple, List, Dict

import torch
from torchaudio.backend.soundfile_backend import load
from torchaudio.transforms import Spectrogram
from torchvision.datasets import DatasetFolder


class AudioTrainDataset(DatasetFolder):
    def __init__(self, root, transform=None) -> None:
        super().__init__(root,
                         loader=load,
                         extensions=(".wav",),
                         transform=transform)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes_list, classes_dict = super().find_classes(directory)
        # do not include background noises for now
        classes_list.remove("_background_noise_")
        classes_dict.pop("_background_noise_")
        return classes_list, classes_dict


class PaddingZeros(object):
    def __init__(self, out_length):
        self.out_length = out_length

    def __call__(self, sample):
        wave, sample_rate = sample
        real_length = wave[0].shape[0]
        diff = self.out_length - real_length
        if diff == 0:
            return sample
        padding = torch.zeros(1, diff)
        return torch.cat([padding, wave], dim=1), sample_rate


class CustomSpectogram(object):
    def __init__(self, n_fft, power):
        self.spec = Spectrogram(n_fft=n_fft, power=power)

    def __call__(self, sample):
        return self.spec(sample[0]), sample[1]
