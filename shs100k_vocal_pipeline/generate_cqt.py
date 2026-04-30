import os
import librosa
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

IN_DIRS = [
    '../data/train/shs100k_train_vocal',
    '../data/test/shs100k_test_vocal',
]
OUT_DIR = 'cqt_features'


def cqt_worker(args):
    in_path, out_path = args
    if os.path.exists(out_path):
        return

    try:
        data, sr = librosa.load(in_path)
        if len(data) < 1000:
            return

        cqt = np.abs(librosa.cqt(y=data, sr=sr))
        mean_size = 20
        height, length = cqt.shape
        if length < mean_size:
            return

        new_cqt = np.zeros((height, int(length / mean_size)), dtype=np.float64)
        for i in range(int(length / mean_size)):
            start = i * mean_size
            end = (i + 1) * mean_size
            new_cqt[:, i] = cqt[:, start:end].mean(axis=1)

        np.save(out_path, new_cqt)
    except Exception as ex:
        print(f'Error processing {in_path}: {ex}')


def collect_jobs():
    params = []
    for in_dir in IN_DIRS:
        if not os.path.isdir(in_dir):
            print(f'Skipping missing directory: {in_dir}')
            continue

        for root, _, files in os.walk(in_dir):
            for filename in files:
                if not filename.endswith('.mp3'):
                    continue
                in_path = os.path.join(root, filename)
                out_path = os.path.join(OUT_DIR, filename.rsplit('.', 1)[0] + '.npy')
                params.append((in_path, out_path))
    return params


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    params = collect_jobs()
    print(f'Found {len(params)} vocal files. Generating CQTs...')

    workers = max(1, min(cpu_count(), 24))
    with Pool(workers) as pool:
        list(tqdm(pool.imap(cqt_worker, params), total=len(params)))

    print('Done generating SHS100k vocal CQT features.')


if __name__ == '__main__':
    main()
