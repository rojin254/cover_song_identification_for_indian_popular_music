"""
Inference script: evaluate cqtnet_shs100k_best.pth on
  1. SHS100K full CQT  (train + test combined)
  2. SHS100K vocal CQT (train + test combined)

Only the embedding layer (fc0 output) is used; fc1 is ignored.
"""

import os
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

# Make sure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import models
from utility import norm, calc_MAP

# ── helpers ────────────────────────────────────────────────────────────────────

def safe_load_model(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location='cpu')
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f'Loaded {len(filtered)} tensors from {checkpoint_path}')
    if missing:
        print(f'  Missing keys : {len(missing)}')
    if unexpected:
        print(f'  Unexpected   : {len(unexpected)}')


def val_combined(model, dataloader, device, dataset_name):
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for data, labels, _ in tqdm(dataloader, desc=f'Evaluating {dataset_name}'):
            data = data.to(device)
            _, embedding = model(data)
            all_embeddings.append(embedding.cpu().numpy())
            all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_labels)
    embeddings = norm(embeddings)
    dis2d = -np.matmul(embeddings, embeddings.T)
    map_score, top10, rank1 = calc_MAP(dis2d, labels)
    print(f'\n[{dataset_name}]  MAP={map_score:.4f}  Top10={top10:.4f}  MRR/Rank1={rank1:.2f}')
    return map_score, top10, rank1


# ── combined dataset (reads two txt files, merges them) ───────────────────────

def pad_or_truncate(data, target_length=400, target_freq=84):
    if data.ndim == 2:
        data = data.unsqueeze(0)
    _, freq, cur_len = data.shape
    if freq > target_freq:
        data = data[:, :target_freq, :]
    elif freq < target_freq:
        data = F.pad(data, (0, 0, 0, target_freq - freq))
    if cur_len > target_length:
        data = data[:, :, :target_length]
    elif cur_len < target_length:
        data = F.pad(data, (0, target_length - cur_len))
    return data


def cut_data_front(data, out_length=None):
    if out_length is not None:
        if data.shape[1] > out_length:
            data = data[:, :out_length]
        else:
            data = np.pad(data, ((0, 0), (0, out_length - data.shape[1])), 'constant')
    if data.shape[1] < 200:
        data = np.pad(data, ((0, 0), (0, 200 - data.shape[1])), 'constant')
    return data


class CombinedCQT(Dataset):
    """Loads two txt-files and combines them into a single dataset."""

    def __init__(self, train_txt, test_txt, cqt_dir):
        self.cqt_dir = cqt_dir
        entries = []
        for txt in (train_txt, test_txt):
            with open(txt) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(line)
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        parts = self.entries[idx].split(',')
        filename = parts[0]
        clique = int(parts[1])
        is_orig = bool(int(parts[2])) if len(parts) > 2 else False

        data = np.load(os.path.join(self.cqt_dir, filename + '.npy'))
        data = data.astype(np.float32) / (np.max(np.abs(data)) + 1e-6)
        data = cut_data_front(data)
        data = torch.tensor(data).unsqueeze(0)  # [1, freq, time]
        data = pad_or_truncate(data)
        return data, clique, is_orig


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = os.path.join(ROOT, 'check_points', 'cqtnet_shs100k_best.pth')
    print(f'Checkpoint : {checkpoint}')
    print(f'Device     : {device}')

    # ── 1. SHS100K full CQT ──────────────────────────────────────────────────
    print('\n' + '='*60)
    print('RUN 1: SHS100K Full CQT  (train + test combined)')
    print('='*60)

    full_train_txt = os.path.join(ROOT, 'shs100k_new_pipeline', 'shs100k_train.txt')
    full_test_txt  = os.path.join(ROOT, 'shs100k_new_pipeline', 'shs100k-test.txt')
    full_cqt_dir   = os.path.join(ROOT, 'shs100k_new_pipeline', 'cqt_features')

    full_ds = CombinedCQT(full_train_txt, full_test_txt, full_cqt_dir)
    print(f'Combined dataset size: {len(full_ds)} tracks')

    full_loader = DataLoader(full_ds, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = models.CQTNet()
    safe_load_model(model, checkpoint)
    model = model.to(device)

    full_map, full_top10, full_rank1 = val_combined(
        model, full_loader, device, 'SHS100K Full CQT (train+test)')

    # ── 2. SHS100K vocal CQT ─────────────────────────────────────────────────
    print('\n' + '='*60)
    print('RUN 2: SHS100K Vocal CQT  (train + test combined)')
    print('='*60)

    vocal_train_txt = os.path.join(ROOT, 'shs100k_vocal_pipeline', 'shs100k_train.txt')
    vocal_test_txt  = os.path.join(ROOT, 'shs100k_vocal_pipeline', 'shs100k-test.txt')
    vocal_cqt_dir   = os.path.join(ROOT, 'shs100k_vocal_pipeline', 'cqt_features')

    vocal_ds = CombinedCQT(vocal_train_txt, vocal_test_txt, vocal_cqt_dir)
    print(f'Combined dataset size: {len(vocal_ds)} tracks')

    vocal_loader = DataLoader(vocal_ds, batch_size=64, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Reuse the same model (already loaded)
    vocal_map, vocal_top10, vocal_rank1 = val_combined(
        model, vocal_loader, device, 'SHS100K Vocal CQT (train+test)')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('SUMMARY  —  cqtnet_shs100k_best.pth')
    print('='*60)
    print(f'{"Dataset":<35} {"MAP":>8} {"Top10":>8} {"Rank1":>8}')
    print('-'*60)
    print(f'{"SHS100K Full CQT (train+test)":<35} {full_map:>8.4f} {full_top10:>8.4f} {full_rank1:>8.2f}')
    print(f'{"SHS100K Vocal CQT (train+test)":<35} {vocal_map:>8.4f} {vocal_top10:>8.4f} {vocal_rank1:>8.2f}')
    print('='*60)


if __name__ == '__main__':
    main()
