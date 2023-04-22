from typing import Tuple, List, Dict

import torch
from torchaudio.backend.soundfile_backend import load
from torchaudio.transforms import Spectrogram, MelSpectrogram
from torchvision.datasets import DatasetFolder


class AudioTrainDataset(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None) -> None:
        super().__init__(root,
                         loader=load,
                         extensions=(".wav",),
                         transform=transform,
                         target_transform=target_transform)

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
        return self.spec(sample[0]).transpose(-1, -2)


class RemoveSampleRate(object):
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        return sample[0][0]

class TargetEncoder:

    def __init__(self, class_dict, commands=None):
        if commands is None:
            commands = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
        self.class_dict = class_dict
        self.commands = commands

    def __call__(self, y):
        class_name = self.class_dict[y]
        if class_name in self.commands:
            y_enc = self.commands.index(class_name)
        elif class_name != "silence":
            y_enc = len(self.commands)
        else:
            y_enc = len(self.commands) + 1
        return torch.nn.functional.one_hot(torch.LongTensor([y_enc]), len(self.commands) + 2).squeeze().to(torch.float)


class NormalizedMelSpectogram:

    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=128):
        self.spec = MelSpectrogram(sample_rate, n_fft=n_fft, hop_length=hop_length, normalized=True)

    def __call__(self, sample, *args, **kwargs):
        return torch.log(self.spec(sample[0]).permute(0, 2, 1))
