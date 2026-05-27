"""
Test vocal-pipeline checkpoints on the non-vocal Indian test set
(i.e., the same test set used by indian_new_pipeline).
"""
import os, sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models
from utility import norm, calc_MAP
from indian_new_pipeline.cqt_loader import IndianCQT

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_VOCAL_CLASSES = 402   # from indian_vocal_pipeline train set
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'check_points')

CHECKPOINTS = {
    'CE (185647)':          'CQTNet_indian_vocal_ce_best_indian_vocal_20260430_185647.pth',
    'CE (201641)':          'CQTNet_indian_vocal_ce_best_indian_vocal_20260430_201641.pth',
    'Triplet (183443)':     'CQTNet_indian_vocal_triplet_best_indian_vocal_triplet_20260430_183443.pth',
    'DynTriplet full':      'CQTNet_indian_vocal_dyn_triplet_best_vocal_dyn_triplet_ce201641_full.pth',
    'DynTriplet ce probe':  'CQTNet_indian_vocal_dyn_triplet_best_vocal_dyn_triplet_ce201641_probe.pth',
    'DynTriplet 183 probe': 'CQTNet_indian_vocal_dyn_triplet_best_vocal_dyn_triplet_183443_probe.pth',
}


def build_model(checkpoint_path):
    model = models.CQTNet()
    model.fc1 = nn.Linear(300, NUM_VOCAL_CLASSES)
    state = torch.load(checkpoint_path, map_location='cpu')
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_state and model_state[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f'  [warn] {len(missing)} missing keys')
    return model.to(DEVICE).eval()


@torch.no_grad()
def evaluate(model, dataloader):
    all_emb, all_labels = [], []
    for data, labels, _ in tqdm(dataloader, desc='  evaluating', leave=False):
        _, emb = model(data.to(DEVICE))
        all_emb.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())
    emb = norm(np.concatenate(all_emb))
    labels = np.concatenate(all_labels)
    dis2d = -np.matmul(emb, emb.T)
    return calc_MAP(dis2d, labels)


def main():
    test_data = IndianCQT('test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False,
                             num_workers=2)

    print(f'\nDevice: {DEVICE}')
    print(f'Test samples: {len(test_data)}\n')
    print(f'{"Checkpoint":<28}  {"MAP":>7}  {"Top10":>7}  {"Rank1":>7}')
    print('-' * 58)

    results = {}
    for name, fname in CHECKPOINTS.items():
        ckpt_path = os.path.join(CHECKPOINT_DIR, fname)
        if not os.path.exists(ckpt_path):
            print(f'{name:<28}  [NOT FOUND]')
            continue
        model = build_model(ckpt_path)
        map_score, top10, rank1 = evaluate(model, test_loader)
        results[name] = (map_score, top10, rank1)
        print(f'{name:<28}  {map_score:>7.4f}  {top10:>7.4f}  {rank1:>7.2f}')

    print()


if __name__ == '__main__':
    main()
