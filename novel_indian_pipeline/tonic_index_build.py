"""Precompute per-file tonic offset for a manifest, write to a sidecar JSON.

Run once per dataset; training reads the JSON and never re-estimates.

Usage:
    python -m novel_indian_pipeline.tonic_index_build --dataset indian
    python -m novel_indian_pipeline.tonic_index_build --dataset indian_vocal
    python -m novel_indian_pipeline.tonic_index_build --dataset shs100k
    python -m novel_indian_pipeline.tonic_index_build --dataset shs100k_vocal

Manifest formats are inferred from `env_paths.DATASETS[name]['fmt']`:
  - 'indian'  -> "<filename>,<clique>" (is_original from suffix)
  - 'shs100k' -> "<filename>,<clique>,<is_original_int>"

Both formats yield the same set of `<filename>` keys to estimate against.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from novel_indian_pipeline.env_paths import DATASETS, dataset_paths, tonic_index_dir
from novel_indian_pipeline.tonic import estimate_tonic_from_cqt


def _read_manifest_filenames(manifest_path: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    with open(manifest_path) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            fname = line.split(',', 1)[0]
            if fname in seen:
                continue
            seen.add(fname)
            out.append(fname)
    return out


def build_index(dataset: str) -> str:
    if dataset not in DATASETS:
        raise SystemExit(f'unknown dataset {dataset!r}')
    paths = dataset_paths(dataset)
    cqt_dir = paths['cqt_dir']

    filenames: list[str] = []
    for key in ('train_txt', 'test_txt'):
        manifest = paths[key]
        if manifest:
            filenames.extend(_read_manifest_filenames(manifest))

    seen: set[str] = set()
    deduped = []
    for f in filenames:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    filenames = deduped

    out_dir = tonic_index_dir()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{dataset}.json')

    index: dict[str, int] = {}
    skipped = 0
    for fname in tqdm(filenames, desc=f'tonic[{dataset}]'):
        npy_path = os.path.join(cqt_dir, fname + '.npy')
        if not os.path.exists(npy_path):
            skipped += 1
            continue
        try:
            arr = np.load(npy_path)
        except Exception as e:  # noqa: BLE001
            print(f'  load failed {fname}: {e}', file=sys.stderr)
            skipped += 1
            continue
        index[fname] = estimate_tonic_from_cqt(arr)

    with open(out_path, 'w') as fp:
        json.dump(index, fp)
    print(f'Wrote {len(index)} tonic entries to {out_path} (skipped {skipped})')
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Build per-file tonic index for a dataset.')
    ap.add_argument('--dataset', required=True, choices=list(DATASETS))
    args = ap.parse_args()
    build_index(args.dataset)


if __name__ == '__main__':
    main()
