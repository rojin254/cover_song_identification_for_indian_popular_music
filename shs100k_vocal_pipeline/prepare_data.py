import os
import pandas as pd


TRAIN_CSV = '../data/shs100k-train.csv'
TEST_CSV = '../data/shs100k-test.csv'
TRAIN_AUDIO_DIR = '../data/train/shs100k_train_vocal'
TEST_AUDIO_DIR = '../data/test/shs100k_test_vocal'
TRAIN_OUT = 'shs100k_train.txt'
TEST_OUT = 'shs100k-test.txt'


def collect_available_ids(audio_dir):
    available = set()
    if not os.path.isdir(audio_dir):
        return available

    for root, _, files in os.walk(audio_dir):
        for filename in files:
            if filename.endswith('.mp3'):
                available.add(filename.rsplit('.', 1)[0])
    return available


def prepare_shs100k(csv_path, out_txt_path, available_ids):
    print(f'Processing {csv_path} ...')
    df = pd.read_csv(csv_path)

    with open(out_txt_path, 'w') as fp:
        for clique, group in df.groupby('clique'):
            min_version = group['version'].min()
            for _, row in group.iterrows():
                filename = row['filename'].rsplit('.', 1)[0]
                if filename not in available_ids:
                    continue

                is_original = 1 if row['version'] == min_version else 0
                fp.write(f'{filename},{clique},{is_original}\n')

    print(f'Saved {out_txt_path}')


def main():
    train_available = collect_available_ids(TRAIN_AUDIO_DIR)
    test_available = collect_available_ids(TEST_AUDIO_DIR)

    prepare_shs100k(TRAIN_CSV, TRAIN_OUT, train_available)
    prepare_shs100k(TEST_CSV, TEST_OUT, test_available)


if __name__ == '__main__':
    main()
