import torch
import numpy as np
import os
from cqt_loader import IndianCoverCQT
from main import custom_collate, create_triplets
from torch.utils.data import DataLoader
from config import opt

# We create some fake npy files for the test
os.makedirs('indian_new_pipeline/cqt_features', exist_ok=True)
fake_ids = ['100-1-ci-c', '100-2-ci-c', '100-0-ci-o', '101-1-ci-c', '101-0-ci-o']

with open('indian_new_pipeline/indian_train.txt', 'w') as f:
    for idx in fake_ids:
        f.write(f'{idx}\n')

for idx in fake_ids:
    np.save(f'indian_new_pipeline/cqt_features/{idx}.npy', np.random.rand(12, 100))

dataset = IndianCoverCQT('train')
dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=custom_collate)

opt.device = 'cpu'

for batch in dataloader:
    data, labels, is_orig = batch
    print(f"Data shape: {data.shape}")
    print(f"Labels: {labels}")
    print(f"Is_orig: {is_orig}")
    
    # Fake embeddings
    embeddings = torch.randn(len(data), 300)
    
    anchors, positives, negatives = create_triplets(embeddings, labels, is_orig)
    print(f"Anchors shape: {anchors.shape}")
    print(f"Positives shape: {positives.shape}")
    print(f"Negatives shape: {negatives.shape}")
    break
