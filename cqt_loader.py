import os
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
from torchvision import transforms
import random
import PIL
import torch.nn.functional as F

class IndianCoverCQT(Dataset):
    def __init__(self, mode='train', out_length=None):
        self.indir = 'indian_new_pipeline/cqt_features'
        self.mode = mode
        if self.mode=='train':
          self.filepath = 'indian_new_pipeline/indian_train.txt'
          self.set = 0
          self.set_dict = {}
        elif self.mode=='val':
          self.filepath = 'indian_new_pipeline/indian_train.txt'
        else:
          self.filepath = 'indian_new_pipeline/indian_test.txt'
        
        with open(self.filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]
        self.out_length = out_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        line_parts = self.file_list[index].strip().split(',')
        filename = line_parts[0]
        set_id = line_parts[1] if len(line_parts) > 1 else filename.split('-')[0]
        if self.mode == 'train':
            if set_id in self.set_dict:
                set_id = self.set_dict[set_id]
            else:
                self.set_dict[set_id] = self.set
                set_id = self.set
                self.set += 1
        in_path = os.path.join(self.indir, filename + '.npy')
        data = np.load(in_path)
        
        transform_test = transforms.Compose([
            lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
            lambda x: self.cut_data_front(x, self.out_length),
            lambda x: torch.Tensor(x),
            lambda x: x.unsqueeze(0),  # Add channel dimension
        ])
        
        data = transform_test(data)
        data = self.pad_or_truncate(data, 400, 84)
        
        is_original = filename.endswith('-o') or '-o-' in filename
        return data, int(set_id), is_original

    # Including necessary methods from the CQT class
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
                data = np.pad(data, ((0, 0), (0, offset)), "constant")
        if data.shape[1] < 200:
            offset = 200 - data.shape[1]
            data = np.pad(data, ((0, 0), (0, offset)), "constant")
        return data

if __name__ == '__main__':
    train_dataset = IndianCoverCQT('train', 394)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=12, shuffle=True)
