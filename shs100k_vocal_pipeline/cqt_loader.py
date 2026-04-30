import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import torch.nn.functional as F


class SHS100kVocalCQT(Dataset):
    def __init__(self, mode='train', out_length=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.indir = os.path.join(base_dir, 'cqt_features')
        self.mode = mode

        if mode == 'train':
            self.filepath = os.path.join(base_dir, 'shs100k_train.txt')
            self.label_count = 0
            self.label_map = {}
        elif mode == 'val':
            self.filepath = os.path.join(base_dir, 'shs100k_train.txt')
        else:
            self.filepath = os.path.join(base_dir, 'shs100k-test.txt')

        with open(self.filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp if line.strip()]

        if mode == 'train':
            for line in self.file_list:
                clique = line.split(',')[1]
                if clique not in self.label_map:
                    self.label_map[clique] = self.label_count
                    self.label_count += 1

        self.out_length = out_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        line_parts = self.file_list[index].split(',')
        filename = line_parts[0]
        clique = line_parts[1]
        is_original = bool(int(line_parts[2])) if len(line_parts) > 2 else False

        if self.mode == 'train':
            clique = self.label_map[clique]
        else:
            clique = int(clique)

        in_path = os.path.join(self.indir, filename + '.npy')
        data = np.load(in_path)

        transform_test = transforms.Compose([
            lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
            lambda x: self.cut_data_front(x, self.out_length),
            lambda x: torch.Tensor(x),
            lambda x: x.unsqueeze(0),
        ])

        data = transform_test(data)
        data = self.pad_or_truncate(data, 400, 84)
        return data, int(clique), is_original

    def pad_or_truncate(self, data, target_length, target_freq=84):
        if data.ndim == 2:
            data = data.unsqueeze(0)

        _, freq, current_length = data.shape

        if freq > target_freq:
            data = data[:, :target_freq, :]
        elif freq < target_freq:
            pad_freq = target_freq - freq
            data = F.pad(data, (0, 0, 0, pad_freq), mode='constant', value=0)

        if current_length > target_length:
            data = data[:, :, :target_length]
        elif current_length < target_length:
            pad_time = target_length - current_length
            data = F.pad(data, (0, pad_time), mode='constant', value=0)

        return data

    def cut_data_front(self, data, out_length):
        if out_length is not None:
            if data.shape[1] > out_length:
                data = data[:, :out_length]
            else:
                offset = out_length - data.shape[1]
                data = np.pad(data, ((0, 0), (0, offset)), 'constant')

        if data.shape[1] < 200:
            offset = 200 - data.shape[1]
            data = np.pad(data, ((0, 0), (0, offset)), 'constant')
        return data
