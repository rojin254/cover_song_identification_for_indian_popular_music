"""
Compute N_sim lookup: for each song, find the most similar song from a
different clique using cosine similarity on raw CQT features.

Outputs nsim_lookup.json mapping each song ID to its hardest non-cover negative.
Runs on CPU only (numpy).
"""

import os
import json
import numpy as np
from tqdm import tqdm

CQT_DIR = 'cqt_features'
TRAIN_TXT = 'indian_train.txt'
TEST_TXT = 'indian_test.txt'
OUTPUT_FILE = 'nsim_lookup.json'


def load_txt(filepath):
    """Load (filename, clique) pairs from a txt file. Deduplicates."""
    entries = {}
    if not os.path.exists(filepath):
        print(f'Skipping missing file: {filepath}')
        return entries
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            filename = parts[0]
            clique = parts[1]
            entries[filename] = clique
    return entries


def preprocess_cqt(data, target_freq=84, target_length=400):
    """Normalize, pad/truncate to consistent shape, then flatten."""
    max_val = np.max(np.abs(data))
    if max_val > 1e-6:
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    freq, length = data.shape

    if freq > target_freq:
        data = data[:target_freq, :]
    elif freq < target_freq:
        data = np.pad(data, ((0, target_freq - freq), (0, 0)), mode='constant')

    if length > target_length:
        data = data[:, :target_length]
    elif length < target_length:
        data = np.pad(data, ((0, 0), (0, target_length - length)), mode='constant')

    return data.flatten()


def main():
    entries = {}
    entries.update(load_txt(TRAIN_TXT))
    entries.update(load_txt(TEST_TXT))
    print(f'Total unique songs: {len(entries)}')

    filenames = []
    cliques = []
    vectors = []
    missing = 0

    for fname, clique in tqdm(entries.items(), desc='Loading CQTs'):
        npy_path = os.path.join(CQT_DIR, fname + '.npy')
        if not os.path.exists(npy_path):
            missing += 1
            continue
        data = np.load(npy_path)
        vec = preprocess_cqt(data)
        filenames.append(fname)
        cliques.append(clique)
        vectors.append(vec)

    if missing:
        print(f'Warning: {missing} files not found in {CQT_DIR}')

    print(f'Loaded {len(vectors)} CQT vectors')

    matrix = np.stack(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    matrix = matrix / norms

    print('Computing cosine similarity matrix...')
    sim_matrix = matrix @ matrix.T

    clique_arr = np.array(cliques)
    lookup = {}

    for i in tqdm(range(len(filenames)), desc='Finding N_sim'):
        sims = sim_matrix[i].copy()
        same_clique_mask = clique_arr == clique_arr[i]
        sims[same_clique_mask] = -np.inf

        best_j = int(np.argmax(sims))
        lookup[filenames[i]] = {
            'nsim_id': filenames[best_j],
            'nsim_clique': cliques[best_j],
            'cosine_similarity': float(sims[best_j]),
        }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(lookup, f, indent=2)

    print(f'Saved N_sim lookup for {len(lookup)} songs to {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
