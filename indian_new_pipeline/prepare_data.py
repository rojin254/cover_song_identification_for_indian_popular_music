import os
import glob
import pandas as pd
import numpy as np
import librosa
from multiprocessing import Pool
from tqdm import tqdm

TRAIN_DIR = '../data/train/indian'
TEST_DIR = '../data/test/indian'
TRAIN_CSV = '../data/indian_train.csv'
TEST_CSV = '../data/indian_test.csv'
OUT_DIR = 'cqt_features'
TRAIN_TXT = 'indian_train.txt'
TEST_TXT = 'indian_test.txt'

def CQT(args):
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
        new_cqt = np.zeros((height, int(length/mean_size)), dtype=np.float64)
        for i in range(int(length/mean_size)):
            new_cqt[:,i] = cqt[:, i*mean_size:(i+1)*mean_size].mean(axis=1)
        np.save(out_path, new_cqt)
    except Exception as e:
        print(f'Error processing {in_path}: {e}')

def generate_txt(csv_file, txt_file):
    df = pd.read_csv(csv_file)
    with open(txt_file, 'w') as f:
        for _, row in df.iterrows():
            idx = row['id']
            clique = row['clique']
            if os.path.exists(os.path.join(OUT_DIR, f"{idx}.npy")):
                f.write(f'{idx},{clique}\n')

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Extract CQT features
    print("Preparing to extract CQT features...")
    params = []
    for d in [TRAIN_DIR, TEST_DIR]:
        for f in os.listdir(d):
            if f.endswith('.mp3'):
                in_path = os.path.join(d, f)
                out_path = os.path.join(OUT_DIR, f.replace('.mp3', '.npy'))
                params.append((in_path, out_path))
                
    print(f"Total files to process: {len(params)}")
    
    with Pool(40) as pool:
        list(tqdm(pool.imap(CQT, params), total=len(params)))

    # 2. Generate text splits
    print("Generating .txt splits...")
    generate_txt(TRAIN_CSV, TRAIN_TXT)
    generate_txt(TEST_CSV, TEST_TXT)

    print("Data preparation complete.")

if __name__ == '__main__':
    main()
