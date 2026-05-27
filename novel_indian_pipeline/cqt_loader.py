"""Generalised CQT loader serving all four datasets.

Reads CQT .npy files from an external directory, applies normalisation,
optional tonic-relative rotation, and pad/truncate to (1, 84, 400). The
manifest format is auto-detected:

  - 'indian'  : "<filename>,<clique>"   is_original = filename suffix '-o'
  - 'shs100k' : "<filename>,<clique>,<is_orig>"

Two operating modes:

  - mode == 'train': build a per-clique label map (clique str -> int);
                     subsequent calls return clique as int.
  - mode == 'val' / 'test': return the clique as parsed from the manifest;
                            for SHS100K it is an int already; for Indian it
                            is the original clique-string-as-int (the
                            existing repo convention).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from novel_indian_pipeline.cqt_rotate import rotate_cqt


def _is_original_indian(filename: str) -> bool:
    return filename.endswith('-o') or '-o-' in filename


class GenericCQT(Dataset):
    """Single loader serving all four datasets in this pipeline.

    Args:
        cqt_dir: directory of pre-computed `<filename>.npy` features.
        manifest_path: path to the manifest .txt file.
        fmt: 'indian' or 'shs100k'.
        mode: 'train' | 'val' | 'test'. In 'train', a label_map is built.
        out_length: optional CQT time crop before pad-to-400.
        tonic_normalize: if True, rotate the CQT so tonic = bin 0.
        tonic_index_path: required iff tonic_normalize is True. JSON
            mapping filename -> tonic offset {0..11}.
        label_map: optional pre-built {clique_str: int}; if provided,
            *forces* this mapping (used for val/test against a known train
            labelling — not used in this codebase but reserved for future).
    """

    def __init__(
        self,
        cqt_dir: str,
        manifest_path: str,
        fmt: str = 'indian',
        mode: str = 'train',
        out_length: int | None = None,
        tonic_normalize: bool = False,
        tonic_index_path: str | None = None,
        label_map: dict | None = None,
    ):
        if fmt not in ('indian', 'shs100k'):
            raise ValueError(f'unknown fmt {fmt!r}')
        if tonic_normalize and not tonic_index_path:
            raise ValueError(
                'tonic_normalize=True requires tonic_index_path; run '
                'tonic_index_build.py first.'
            )

        self.cqt_dir = cqt_dir
        self.manifest_path = manifest_path
        self.fmt = fmt
        self.mode = mode
        self.out_length = out_length
        self.tonic_normalize = tonic_normalize

        with open(manifest_path) as fp:
            self.file_list = [line.rstrip() for line in fp if line.strip()]

        # Build / accept label map (only meaningful for 'train').
        if mode == 'train':
            if label_map is None:
                self.label_map: dict = {}
                self.label_count = 0
                for line in self.file_list:
                    clique = line.split(',')[1]
                    if clique not in self.label_map:
                        self.label_map[clique] = self.label_count
                        self.label_count += 1
            else:
                self.label_map = dict(label_map)
                self.label_count = max(self.label_map.values()) + 1 if self.label_map else 0
        else:
            self.label_map = label_map or {}
            self.label_count = max(self.label_map.values()) + 1 if self.label_map else 0

        if tonic_normalize:
            with open(tonic_index_path) as fp:
                self.tonic_index: dict = json.load(fp)
        else:
            self.tonic_index = {}

    def __len__(self) -> int:
        return len(self.file_list)

    def _parse(self, line: str) -> tuple[str, str, bool]:
        parts = line.strip().split(',')
        filename = parts[0]
        clique = parts[1]
        if self.fmt == 'indian':
            is_orig = _is_original_indian(filename)
        else:  # shs100k
            is_orig = bool(int(parts[2])) if len(parts) > 2 else False
        return filename, clique, is_orig

    def __getitem__(self, index: int):
        filename, clique, is_original = self._parse(self.file_list[index])

        if self.mode == 'train' and clique in self.label_map:
            clique_out: int = self.label_map[clique]
        else:
            try:
                clique_out = int(clique)
            except ValueError:
                clique_out = self.label_map.get(clique, -1)

        in_path = os.path.join(self.cqt_dir, filename + '.npy')
        data = np.load(in_path)

        if self.tonic_normalize:
            tonic = self.tonic_index.get(filename, 0)
            data = rotate_cqt(data, tonic)

        transform_test = transforms.Compose([
            lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
            lambda x: self._cut_data_front(x, self.out_length),
            lambda x: torch.Tensor(x),
            lambda x: x.unsqueeze(0),
        ])
        data = transform_test(data)
        data = self._pad_or_truncate(data, 400, 84)
        return data, int(clique_out), is_original

    @staticmethod
    def _pad_or_truncate(data, target_length, target_freq=84):
        if data.ndim == 2:
            data = data.unsqueeze(0)
        _, freq, current_length = data.shape
        if freq > target_freq:
            data = data[:, :target_freq, :]
        elif freq < target_freq:
            data = F.pad(data, (0, 0, 0, target_freq - freq), mode='constant', value=0)
        if current_length > target_length:
            data = data[:, :, :target_length]
        elif current_length < target_length:
            data = F.pad(data, (0, target_length - current_length), mode='constant', value=0)
        return data

    @staticmethod
    def _cut_data_front(data, out_length):
        if out_length is not None:
            if data.shape[1] > out_length:
                data = data[:, :out_length]
            else:
                data = np.pad(data, ((0, 0), (0, out_length - data.shape[1])), 'constant')
        if data.shape[1] < 200:
            data = np.pad(data, ((0, 0), (0, 200 - data.shape[1])), 'constant')
        return data
