import os
import pandas as pd
import numpy as np
import librosa
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

TRAIN_DIR = '../data/train/indian_vocal_train'
TEST_DIR = '../data/test/indian_vocal'
TRAIN_CSV = '../data/indian_train.csv'
TEST_CSV = '../data/indian_test.csv'
OUT_DIR = 'cqt_features'
TRAIN_TXT = 'indian_train.txt'
TEST_TXT = 'indian_test.txt'


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


def generate_txt(csv_file, txt_file, available_ids):
    df = pd.read_csv(csv_file)
    with open(txt_file, 'w') as fp:
        for _, row in df.iterrows():
            idx = str(row['id'])
            clique = row['clique']
            if idx in available_ids:
                fp.write(f'{idx},{clique}\n')


def collect_audio_jobs():
    params = []
    for source_dir in [TRAIN_DIR, TEST_DIR]:
        if not os.path.isdir(source_dir):
            print(f'Skipping missing directory: {source_dir}')
            continue

        for root, _, files in os.walk(source_dir):
            for filename in files:
                if not filename.endswith('.mp3'):
                    continue
                in_path = os.path.join(root, filename)
                out_path = os.path.join(OUT_DIR, filename.replace('.mp3', '.npy'))
                params.append((in_path, out_path))
    return params


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    params = collect_audio_jobs()
    print(f'Total vocal files to process: {len(params)}')

    workers = max(1, min(cpu_count(), 24))
    with Pool(workers) as pool:
        list(tqdm(pool.imap(cqt_worker, params), total=len(params)))

    available_ids = {
        filename[:-4]
        for filename in os.listdir(OUT_DIR)
        if filename.endswith('.npy')
    }

    generate_txt(TRAIN_CSV, TRAIN_TXT, available_ids)
    generate_txt(TEST_CSV, TEST_TXT, available_ids)

    print('Indian vocal data preparation complete.')


if __name__ == '__main__':
    main()
