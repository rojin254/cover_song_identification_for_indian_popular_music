import librosa
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm

in_dirs = [
    '/teamspace/studios/this_studio/PP4/data/train/shs100k_train',
    '/teamspace/studios/this_studio/PP4/data/test/shs100k-test'
]
out_dir = '/teamspace/studios/this_studio/PP4/shs100k_new_pipeline/cqt_features/'

def CQT(args):
    try:
        in_path, out_path = args
        # Only process if not exists
        if os.path.exists(out_path):
            return
            
        data, sr = librosa.load(in_path)
        if len(data) < 1000:
            return
            
        cqt = np.abs(librosa.cqt(y=data, sr=sr))
        mean_size = 20
        height, length = cqt.shape
        new_cqt = np.zeros((height, int(length / mean_size)), dtype=np.float64)
        for i in range(int(length / mean_size)):
            new_cqt[:, i] = cqt[:, i * mean_size:(i + 1) * mean_size].mean(axis=1)
            
        np.save(out_path, new_cqt)
    except Exception as e:
        print('Error processing:', in_path, e)

if __name__ == '__main__':
    os.makedirs(out_dir, exist_ok=True)
    
    params = []
    for d in in_dirs:
        for root, dirs, files in os.walk(d):  
            for file in files:
                if file.endswith('.mp3'):
                    in_path = os.path.join(root, file)
                    filename = file.rsplit('.', 1)[0]
                    out_path = os.path.join(out_dir, filename + '.npy')
                    params.append((in_path, out_path))

    print(f'Found {len(params)} files. Starting CQT generation...')
    pool = Pool(12)  # Adjust based on available CPU cores
    for _ in tqdm(pool.imap_unordered(CQT, params), total=len(params)):
        pass
    pool.close()
    pool.join()
    print('Done generating CQT features.')
