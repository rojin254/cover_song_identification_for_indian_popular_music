"""Training + evaluation entrypoint for the novel Indian CSI pipeline.

Usage examples:

    # Row 1 baseline reproduction:
    python -m novel_indian_pipeline.main train \\
        --dataset=indian_vocal --row_name=row1_baseline_dyn_triplet

    # Row 7 full recipe:
    python -m novel_indian_pipeline.main train \\
        --dataset=indian_vocal --row_name=row7_full \\
        --tonic_normalize=True --use_supcon=True --use_arcface=True

    # Eval-only on a different dataset using a trained ckpt:
    python -m novel_indian_pipeline.main evaluate \\
        --dataset=shs100k_vocal --checkpoint=/path/to/best.pth

The 8-row ablation matrix is driven by run_ablations.sh.
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)  # so PP4/utility.py resolves

# Package-relative imports keep us safe from PP4/cqt_loader.py shadowing.
from novel_indian_pipeline import models
from novel_indian_pipeline.cqt_loader import GenericCQT
from novel_indian_pipeline.env_paths import (
    DATASETS,
    dataset_paths,
    logs_dir,
    new_ckpt_dir,
    results_dir,
    tonic_index_dir,
)
from novel_indian_pipeline.losses import (
    SubCenterArcFaceHead,
    modified_triplet_loss,
    supcon_loss,
)
from utility import calc_MAP, norm  # reused from PP4 root


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# ---------------------------------------------------------------------------
# DataLoader plumbing.
# ---------------------------------------------------------------------------
def custom_collate(batch):
    data = torch.stack([item[0] for item in batch])
    labels = torch.LongTensor([item[1] for item in batch])
    is_orig = torch.BoolTensor([item[2] for item in batch])
    return data, labels, is_orig


def safe_load_model(model: nn.Module, checkpoint_path: str):
    state = torch.load(checkpoint_path, map_location='cpu')
    own = model.state_dict()
    filt = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(filt, strict=False)
    print(f'Loaded {len(filt)} tensors from {checkpoint_path}')


def _tonic_index_path(dataset: str, tonic_normalize: bool) -> str | None:
    if not tonic_normalize:
        return None
    p = os.path.join(tonic_index_dir(), f'{dataset}.json')
    if not os.path.exists(p):
        raise FileNotFoundError(
            f'Tonic index missing: {p}. Run tonic_index_build.py --dataset={dataset} first.'
        )
    return p


def build_dataset(dataset: str, mode: str, tonic_normalize: bool, label_map=None):
    spec = dataset_paths(dataset)
    manifest = spec['train_txt'] if mode == 'train' else (
        spec['train_txt'] if mode == 'val' else spec['test_txt']
    )
    if manifest is None:
        return None
    return GenericCQT(
        cqt_dir=spec['cqt_dir'],
        manifest_path=manifest,
        fmt=spec['fmt'],
        mode=mode,
        out_length=None,
        tonic_normalize=tonic_normalize,
        tonic_index_path=_tonic_index_path(dataset, tonic_normalize),
        label_map=label_map,
    )


# ---------------------------------------------------------------------------
# Triplet construction (mirrors indian_new_pipeline/main.py).
# ---------------------------------------------------------------------------
def create_modified_triplets(embeddings, labels, is_orig, nsim_emb_dict, device):
    anchors, positives, neg_rands, neg_sims = [], [], [], []
    unique_labels = labels.unique()
    for label in unique_labels:
        label_val = label.item()
        label_mask = labels == label
        original_idx = torch.where(label_mask & is_orig)[0]
        cover_idx = torch.where(label_mask & (~is_orig))[0]
        negative_idx = torch.where(labels != label)[0]
        if len(original_idx) == 0 or len(cover_idx) == 0 or len(negative_idx) == 0:
            continue
        for a_idx in original_idx:
            pos_idx = cover_idx[torch.randint(0, len(cover_idx), (1,)).item()]
            neg_r_idx = negative_idx[torch.randint(0, len(negative_idx), (1,)).item()]
            anchors.append(embeddings[a_idx].unsqueeze(0))
            positives.append(embeddings[pos_idx].unsqueeze(0))
            neg_rands.append(embeddings[neg_r_idx].unsqueeze(0))
            if label_val in nsim_emb_dict:
                neg_sims.append(nsim_emb_dict[label_val].to(embeddings.device).unsqueeze(0))
            else:
                neg_sims.append(embeddings[neg_r_idx].unsqueeze(0))
    if not anchors:
        empty = torch.empty(0, embeddings.shape[1], device=device)
        return empty, empty, empty, empty
    return (
        torch.cat(anchors).to(device),
        torch.cat(positives).to(device),
        torch.cat(neg_rands).to(device),
        torch.cat(neg_sims).to(device),
    )


@torch.no_grad()
def recompute_nsim_from_model(model, train_loader, device):
    """Dynamic N_sim refresh — verbatim logic from the baseline pipeline."""
    model.eval()
    all_embeddings, all_labels, all_is_orig = [], [], []
    for data, labels, is_orig in tqdm(train_loader, desc='nsim-refresh'):
        data = data.to(device)
        _, embedding = model(data)
        all_embeddings.append(embedding.detach().cpu())
        all_labels.append(labels.cpu())
        all_is_orig.append(is_orig.cpu())
    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)
    is_orig = torch.cat(all_is_orig)
    emb_norm = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = emb_norm @ emb_norm.T

    nsim_emb_dict = {}
    orig_indices = torch.where(is_orig)[0]
    for idx in orig_indices:
        sims = sim_matrix[idx].clone()
        same_clique_mask = labels == labels[idx]
        sims[same_clique_mask] = -2.0
        best_neg_idx = sims.argmax().item()
        nsim_emb_dict[labels[idx].item()] = embeddings[best_neg_idx]
    return nsim_emb_dict


# ---------------------------------------------------------------------------
# Evaluation.
# ---------------------------------------------------------------------------
@torch.no_grad()
def val_slow(model, dataloader, device, tag):
    model.eval()
    all_embeddings, all_labels = [], []
    for data, labels, _ in tqdm(dataloader, desc=f'eval[{tag}]'):
        data = data.to(device)
        _, embedding = model(data)
        all_embeddings.append(embedding.cpu().numpy())
        all_labels.append(labels.numpy())
    if not all_embeddings:
        return 0.0, 0.0, 0.0, 0
    embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_labels)
    embeddings = norm(embeddings)
    dis2d = -np.matmul(embeddings, embeddings.T)
    map_score, top10, rank1 = calc_MAP(dis2d, labels)
    return float(map_score), float(top10), float(rank1), int(len(labels))


def evaluate_all_test_sets(model, device, tonic_normalize: bool, row_dir: str):
    """Evaluate the trained model on all four test manifests and write CSVs."""
    os.makedirs(row_dir, exist_ok=True)
    summary_rows = []
    for ds in ('indian', 'indian_vocal', 'shs100k', 'shs100k_vocal'):
        try:
            test_data = build_dataset(ds, 'test', tonic_normalize=tonic_normalize)
        except (FileNotFoundError, RuntimeError) as e:
            print(f'  skip {ds}: {e}')
            summary_rows.append((ds, None, None, None, None, str(e)))
            continue
        if test_data is None:
            continue
        loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2,
                            collate_fn=custom_collate)
        m, t10, r1, n = val_slow(model, loader, device, ds)
        out_csv = os.path.join(row_dir, f'{ds}_test.csv')
        with open(out_csv, 'w', newline='') as fp:
            w = csv.writer(fp)
            w.writerow(['dataset', 'MAP', 'Top10', 'MR1', 'n_queries'])
            w.writerow([ds, f'{m:.4f}', f'{t10:.4f}', f'{r1:.2f}', n])
        print(f'  {ds}: MAP={m:.4f} Top10={t10:.4f} MR1={r1:.2f} n={n}')
        summary_rows.append((ds, m, t10, r1, n, ''))

    summary_csv = os.path.join(row_dir, 'all_datasets_summary.csv')
    with open(summary_csv, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['dataset', 'MAP', 'Top10', 'MR1', 'n_queries', 'note'])
        for r in summary_rows:
            w.writerow(r)
    return summary_rows


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------
def train(
    dataset: str = 'indian_vocal',
    row_name: str = '',
    tonic_normalize: bool = False,
    use_dyn_triplet: bool = True,
    use_supcon: bool = False,
    use_arcface: bool = False,
    supcon_weight: float = 0.5,
    arcface_weight: float = 0.5,
    triplet_margin: float = 0.3,
    supcon_temperature: float = 0.07,
    arcface_K: int = 3,
    arcface_margin: float = 0.3,
    arcface_scale: float = 30.0,
    num_ce_epochs: int = 0,
    num_dyn_epochs: int = 150,
    nsim_refresh_interval: int = 3,
    batch_size: int = 64,
    num_workers: int = 2,
    lr: float = 5e-5,
    patience: int = 25,
    min_delta: float = 0.002,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; dataset: {dataset}; row: {row_name}')

    if not row_name:
        row_name = f'{dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    spec = dataset_paths(dataset)
    if spec['train_txt'] is None:
        raise SystemExit(f'Dataset {dataset!r} has no train_txt; cannot train on it.')

    train_data = build_dataset(dataset, 'train', tonic_normalize=tonic_normalize)
    val_data = build_dataset(dataset, 'val', tonic_normalize=tonic_normalize)
    test_data = build_dataset(dataset, 'test', tonic_normalize=tonic_normalize)
    num_classes = train_data.label_count
    print(f'{dataset} train cliques: {num_classes}')

    model = models.CQTNet()
    model.fc1 = nn.Linear(300, num_classes)
    safe_load_model(model, spec['ckpt'])
    model = model.to(device)

    for name, p in model.named_parameters():
        p.requires_grad = 'conv0' not in name

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=custom_collate,
    )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False,
                            num_workers=1, collate_fn=custom_collate)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False,
                             num_workers=1, collate_fn=custom_collate)

    ckpt_dir = new_ckpt_dir()
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = logs_dir()
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'{row_name}_{dataset}_{timestamp}'
    best_path = os.path.join(ckpt_dir, f'CQTNet_{run_name}_best.pth')
    log_path = os.path.join(log_dir, f'{run_name}.json')
    print(f'Best ckpt -> {best_path}')
    print(f'Log JSON  -> {log_path}')

    run_summary = {
        'run_name': run_name,
        'row_name': row_name,
        'dataset': dataset,
        'flags': {
            'tonic_normalize': tonic_normalize,
            'use_dyn_triplet': use_dyn_triplet,
            'use_supcon': use_supcon,
            'use_arcface': use_arcface,
            'supcon_weight': supcon_weight,
            'arcface_weight': arcface_weight,
            'triplet_margin': triplet_margin,
            'supcon_temperature': supcon_temperature,
            'arcface_K': arcface_K,
            'arcface_margin': arcface_margin,
            'arcface_scale': arcface_scale,
            'num_ce_epochs': num_ce_epochs,
            'num_dyn_epochs': num_dyn_epochs,
            'batch_size': batch_size,
            'lr': lr,
        },
        'history': [],
    }

    arcface_head = None
    if use_arcface:
        arcface_head = SubCenterArcFaceHead(
            num_classes=num_classes, embed_dim=300,
            K=arcface_K, margin=arcface_margin, scale=arcface_scale,
        ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if arcface_head is not None:
        params += list(arcface_head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # ----- Optional CE warm-up (default 0; warm-start ckpts already trained CE).
    if num_ce_epochs > 0:
        ce_criterion = nn.CrossEntropyLoss()
        for epoch in range(num_ce_epochs):
            model.train()
            total = 0.0
            for inputs, labels, _ in tqdm(train_loader, desc=f'CE {epoch + 1}/{num_ce_epochs}'):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                scores, _ = model(inputs)
                loss = ce_criterion(scores, labels)
                loss.backward()
                optimizer.step()
                total += loss.item()
            print(f'CE epoch {epoch + 1}: avg loss {total / max(1, len(train_loader)):.4f}')

    # ----- Main stage: dynamic-modified-triplet (+ optional SupCon + ArcFace).
    if not use_dyn_triplet:
        # Sanity row 8: just save the warm-started/CE-trained model and eval.
        torch.save(model.state_dict(), best_path)
        run_summary['best_val_map'] = None
        with open(log_path, 'w') as fp:
            json.dump(run_summary, fp, indent=2)
    else:
        nsim_emb_dict = recompute_nsim_from_model(model, train_loader, device)
        print(f'Initial dynamic N_sim: {len(nsim_emb_dict)} cliques')

        best_val_map = -1.0
        early_stop = 0

        for epoch in range(num_dyn_epochs):
            if epoch > 0 and epoch % nsim_refresh_interval == 0:
                nsim_emb_dict = recompute_nsim_from_model(model, train_loader, device)

            model.train()
            if arcface_head is not None:
                arcface_head.train()
            total_loss = 0.0
            total_trip = 0.0
            total_sup = 0.0
            total_arc = 0.0
            steps = 0

            for inputs, labels, is_orig in tqdm(
                train_loader, desc=f'Stage2 {epoch + 1}/{num_dyn_epochs}'
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)
                is_orig = is_orig.to(device)

                _, embeddings = model(inputs)

                anchor, positive, neg_rand, neg_sim = create_modified_triplets(
                    embeddings, labels, is_orig, nsim_emb_dict, device
                )
                if anchor.numel() == 0:
                    continue

                trip = modified_triplet_loss(anchor, positive, neg_rand, neg_sim, triplet_margin)
                loss = trip
                total_trip += trip.item()

                if use_supcon:
                    sup = supcon_loss(embeddings, labels, supcon_temperature)
                    loss = loss + supcon_weight * sup
                    total_sup += float(sup.item())

                if use_arcface and arcface_head is not None:
                    arc_logits = arcface_head(embeddings, labels)
                    arc = F.cross_entropy(arc_logits, labels)
                    loss = loss + arcface_weight * arc
                    total_arc += float(arc.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1

            avg_loss = total_loss / max(1, steps)
            print(
                f'Stage2 epoch {epoch + 1}: loss {avg_loss:.4f}'
                f' (trip {total_trip / max(1, steps):.4f}'
                f', sup {total_sup / max(1, steps):.4f}'
                f', arc {total_arc / max(1, steps):.4f})'
            )

            val_map, val_t10, val_r1, _ = val_slow(model, val_loader, device, 'val')
            print(f'  val MAP={val_map:.4f} Top10={val_t10:.4f} MR1={val_r1:.2f}')

            run_summary['history'].append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'val_map': val_map,
                'val_top10': val_t10,
                'val_rank1': val_r1,
            })
            with open(log_path, 'w') as fp:
                json.dump(run_summary, fp, indent=2)

            if val_map > best_val_map + min_delta:
                best_val_map = val_map
                torch.save(model.state_dict(), best_path)
                run_summary['best_val_map'] = best_val_map
                run_summary['best_epoch'] = epoch + 1
                early_stop = 0
                print(f'  new best {best_val_map:.4f} -> {best_path}')
            else:
                early_stop += 1
                if val_map > best_val_map:
                    best_val_map = val_map
                    torch.save(model.state_dict(), best_path)
                    run_summary['best_val_map'] = best_val_map
                    run_summary['best_epoch'] = epoch + 1
                if early_stop >= patience:
                    print('Early stopping.')
                    break

        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))

    # ----- Final eval on all four test manifests.
    print('\n=== Final evaluation on all four test sets ===')
    row_dir = os.path.join(results_dir(), 'tables', f'{row_name}__trainedon_{dataset}')
    summary_rows = evaluate_all_test_sets(
        model, device, tonic_normalize=tonic_normalize, row_dir=row_dir,
    )
    run_summary['final_eval'] = [
        {'dataset': r[0], 'MAP': r[1], 'Top10': r[2], 'MR1': r[3], 'n': r[4], 'note': r[5]}
        for r in summary_rows
    ]
    with open(log_path, 'w') as fp:
        json.dump(run_summary, fp, indent=2)
    print(f'Done. Results in {row_dir}')


def evaluate(
    dataset: str,
    checkpoint: str,
    tonic_normalize: bool = False,
    row_dir: str = '',
):
    """Eval-only entrypoint. Loads `checkpoint`, evaluates on all four test sets."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)

    spec = dataset_paths(dataset)
    if spec['train_txt'] is not None:
        train_data = build_dataset(dataset, 'train', tonic_normalize=tonic_normalize)
        num_classes = train_data.label_count
    else:
        # SHS100K eval-only: pick num_classes from checkpoint shape.
        state = torch.load(checkpoint, map_location='cpu')
        num_classes = state.get('fc1.weight', torch.empty(10000, 300)).shape[0]

    model = models.CQTNet()
    model.fc1 = nn.Linear(300, num_classes)
    safe_load_model(model, checkpoint)
    model = model.to(device)

    if not row_dir:
        row_dir = os.path.join(
            results_dir(), 'tables',
            f'eval_{os.path.splitext(os.path.basename(checkpoint))[0]}',
        )
    evaluate_all_test_sets(model, device, tonic_normalize=tonic_normalize, row_dir=row_dir)


def build_summary():
    """Aggregate per-row CSVs into results/tables/summary.csv + ablation_matrix.md."""
    base = os.path.join(results_dir(), 'tables')
    if not os.path.exists(base):
        print(f'No tables dir at {base}')
        return
    rows = []
    for entry in sorted(os.listdir(base)):
        entry_dir = os.path.join(base, entry)
        if not os.path.isdir(entry_dir):
            continue
        summary_csv = os.path.join(entry_dir, 'all_datasets_summary.csv')
        if not os.path.exists(summary_csv):
            continue
        with open(summary_csv) as fp:
            r = csv.reader(fp)
            header = next(r, None)
            for row in r:
                rows.append([entry] + row)
    out_csv = os.path.join(base, 'summary.csv')
    with open(out_csv, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['row', 'dataset', 'MAP', 'Top10', 'MR1', 'n_queries', 'note'])
        for r in rows:
            w.writerow(r)
    print(f'Wrote {out_csv} ({len(rows)} rows)')

    # Markdown table
    md_path = os.path.join(results_dir(), 'ablation_matrix.md')
    with open(md_path, 'w') as fp:
        fp.write('# Ablation matrix — auto-generated\n\n')
        fp.write('| Row | Dataset | MAP | Top10 | MR1 | n_queries |\n')
        fp.write('|-----|---------|-----|-------|-----|-----------|\n')
        for r in rows:
            fp.write('| ' + ' | '.join(str(x) for x in r[:6]) + ' |\n')
    print(f'Wrote {md_path}')


if __name__ == '__main__':
    import fire
    fire.Fire({'train': train, 'evaluate': evaluate, 'build_summary': build_summary})
