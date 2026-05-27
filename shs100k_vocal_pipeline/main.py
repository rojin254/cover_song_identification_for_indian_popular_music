import os
import sys
import random
from datetime import datetime
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import models
from config import opt
from utility import norm, calc_MAP
from shs100k_vocal_pipeline.cqt_loader import SHS100kVocalCQT


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def custom_collate(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    is_orig = [item[2] for item in batch]
    data = torch.stack(data)
    labels = torch.LongTensor(labels)
    is_orig = torch.BoolTensor(is_orig)
    return data, labels, is_orig


def safe_load_model(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location='cpu')
    model_state = model.state_dict()
    filtered = {
        k: v
        for k, v in state.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f'Loaded {len(filtered)} tensors from {checkpoint_path}')
    if missing:
        print(f'Missing keys count: {len(missing)}')
    if unexpected:
        print(f'Unexpected keys count: {len(unexpected)}')


def create_triplets(embeddings, labels, is_orig, device):
    anchors = []
    positives = []
    negatives = []

    unique_labels = labels.unique()
    for label in unique_labels:
        label_mask = labels == label
        original_idx = torch.where(label_mask & is_orig)[0]
        cover_idx = torch.where(label_mask & (~is_orig))[0]
        negative_idx = torch.where(labels != label)[0]

        if len(original_idx) == 0 or len(cover_idx) == 0 or len(negative_idx) == 0:
            continue

        for anchor_idx in original_idx:
            pos_idx = cover_idx[torch.randint(0, len(cover_idx), (1,)).item()]
            neg_idx = negative_idx[torch.randint(0, len(negative_idx), (1,)).item()]

            anchors.append(embeddings[anchor_idx].unsqueeze(0))
            positives.append(embeddings[pos_idx].unsqueeze(0))
            negatives.append(embeddings[neg_idx].unsqueeze(0))

    if not anchors:
        empty = torch.empty(0, embeddings.shape[1], device=device)
        return empty, empty, empty

    return torch.cat(anchors).to(device), torch.cat(positives).to(device), torch.cat(negatives).to(device)


def modified_triplet_loss(anchor, positive, neg_rand, neg_sim, margin=0.3):
    d_pos = 1 - F.cosine_similarity(anchor, positive, dim=-1)
    d_rand = 1 - F.cosine_similarity(anchor, neg_rand, dim=-1)
    d_sim = 1 - F.cosine_similarity(anchor, neg_sim, dim=-1)
    loss = torch.clamp(margin + 2 * d_pos - (d_sim + d_rand), min=0.0)
    return loss.mean()


def create_modified_triplets(embeddings, labels, is_orig, nsim_emb_dict, device):
    anchors = []
    positives = []
    neg_rands = []
    neg_sims = []

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


def load_nsim_lookup(lookup_path, train_txt_path):
    with open(lookup_path) as f:
        lookup = json.load(f)

    filename_to_clique = {}
    with open(train_txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            filename_to_clique[parts[0]] = parts[1]

    clique_to_original = {}
    for fname, clique in filename_to_clique.items():
        # SHS100k: is_original is the 3rd column (1=orig, 0=cover)
        pass
    # Re-read with is_original flag
    with open(train_txt_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3 and parts[2] == '1':
                clique_to_original[parts[1]] = parts[0]

    nsim_map = {}
    for orig_fname in clique_to_original.values():
        if orig_fname in lookup:
            nsim_map[orig_fname] = lookup[orig_fname]['nsim_id']

    return nsim_map, filename_to_clique, clique_to_original


def compute_nsim_embeddings(model, nsim_map, clique_to_original, cqt_dir, label_map, device):
    nsim_emb_dict = {}
    model.eval()
    with torch.no_grad():
        for clique, orig_fname in clique_to_original.items():
            nsim_fname = nsim_map.get(orig_fname)
            if nsim_fname is None:
                continue
            npy_path = os.path.join(cqt_dir, nsim_fname + '.npy')
            if not os.path.exists(npy_path):
                continue
            data = np.load(npy_path)
            max_val = np.max(np.abs(data))
            if max_val > 1e-6:
                data = data.astype(np.float32) / max_val
            else:
                data = data.astype(np.float32)
            data = torch.Tensor(data).unsqueeze(0)
            if data.ndim == 3:
                _, freq, length = data.shape
                if freq > 84:
                    data = data[:, :84, :]
                elif freq < 84:
                    data = F.pad(data, (0, 0, 0, 84 - freq))
                if length > 400:
                    data = data[:, :, :400]
                elif length < 400:
                    data = F.pad(data, (0, 400 - length))
            data = data.unsqueeze(0).to(device)
            _, emb = model(data)
            if clique in label_map:
                nsim_emb_dict[label_map[clique]] = emb.squeeze(0).detach().cpu()
    return nsim_emb_dict


@torch.no_grad()
def val_slow(model, dataloader, device, dataset_name):
    model.eval()
    all_embeddings = []
    all_labels = []

    for data, labels, _ in tqdm(dataloader, desc=f'Evaluating {dataset_name}'):
        data = data.to(device)
        _, embedding = model(data)
        all_embeddings.append(embedding.cpu().numpy())
        all_labels.append(labels.numpy())

    if not all_embeddings:
        return 0.0, 0.0, 0.0

    embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_labels)
    embeddings = norm(embeddings)
    dis2d = -np.matmul(embeddings, embeddings.T)
    map_score, top10, rank1 = calc_MAP(dis2d, labels)

    print(f'\nResults for {dataset_name}: MAP={map_score:.4f}, Top10={top10:.4f}, Rank1={rank1:.2f}')
    return map_score, top10, rank1


def transfer_learning(**kwargs):
    opt.batch_size = 120
    opt.num_workers = 4
    opt.model = 'CQTNet'
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt._parse(kwargs)
    print(f'Using device: {opt.device}')

    train_data = SHS100kVocalCQT('train')
    val_data = SHS100kVocalCQT('val')
    test_data = SHS100kVocalCQT('test')

    num_classes = train_data.label_count
    print(f'SHS100k vocal train classes: {num_classes}')

    model = getattr(models, opt.model)()
    model.fc1 = nn.Linear(300, num_classes)

    default_ckpt = os.path.join(parent_dir, 'check_points', 'cqtnet_shs100k_best.pth')
    load_path = kwargs.get('load_model_path', default_ckpt)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f'Checkpoint not found: {load_path}')

    safe_load_model(model, load_path)
    model = model.to(opt.device)

    for name, param in model.named_parameters():
        param.requires_grad = 'conv0' not in name

    train_loader = DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    checkpoint_dir = os.path.join(parent_dir, 'check_points')
    os.makedirs(checkpoint_dir, exist_ok=True)
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    run_name = kwargs.get('run_name')
    if not run_name:
        run_name = f"shs100k_vocal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_log_path = os.path.join(logs_dir, f'{run_name}.json')
    run_summary = {
        'run_name': run_name,
        'dataset': 'shs100k_vocal',
        'checkpoint_init': load_path,
        'batch_size': opt.batch_size,
        'ce_history': [],
        'ce_best_val_map': None,
        'ce_best_epoch': None,
        'ce_checkpoint': None,
        'ce_test_map': None,
        'ce_test_top10': None,
        'ce_test_rank1': None,
        'triplet_history': [],
        'triplet_best_val_map': None,
        'triplet_best_epoch': None,
        'triplet_checkpoint': None,
        'triplet_test_map': None,
        'triplet_test_top10': None,
        'triplet_test_rank1': None,
        'mod_triplet_history': [],
        'mod_triplet_best_val_map': None,
        'mod_triplet_best_epoch': None,
        'mod_triplet_checkpoint': None,
        'mod_triplet_test_map': None,
        'mod_triplet_test_top10': None,
        'mod_triplet_test_rank1': None,
    }

    skip_transfer = kwargs.get('skip_transfer', False)
    num_ce_epochs = 100 if not skip_transfer else 0
    best_val_map = -1.0
    best_ce_path = os.path.join(checkpoint_dir, f'CQTNet_shs100k_vocal_ce_best_{run_name}.pth')
    early_stop = 0
    early_stop_patience = 25
    min_delta = 0.002

    if skip_transfer:
        best_ce_path = load_path
        print('Skipping cross-entropy transfer stage; using init checkpoint directly.')
    else:
        print('\n=== Stage 1: Cross-Entropy Transfer Learning ===')

    for epoch in range(num_ce_epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels, is_orig in tqdm(train_loader, desc=f'CE Epoch {epoch + 1}/{num_ce_epochs}'):
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)
            optimizer.zero_grad()
            scores, _ = model(inputs)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'CE Epoch {epoch + 1}/{num_ce_epochs}, Loss: {avg_loss:.4f}')

        val_map, val_top10, val_rank1 = val_slow(model, val_loader, opt.device, 'SHS100k Vocal Validation')
        print(f'Validation - MAP: {val_map:.4f}, Top10: {val_top10:.4f}, Rank1: {val_rank1:.2f}')

        run_summary['ce_history'].append({
            'epoch': epoch + 1,
            'ce_loss': float(avg_loss),
            'val_map': float(val_map),
            'val_top10': float(val_top10),
            'val_rank1': float(val_rank1),
        })

        if val_map > best_val_map + min_delta:
            best_val_map = val_map
            torch.save(model.state_dict(), best_ce_path)
            run_summary['ce_best_val_map'] = float(best_val_map)
            run_summary['ce_best_epoch'] = epoch + 1
            run_summary['ce_checkpoint'] = best_ce_path
            print(f'New best CE model saved/updated: {best_ce_path}')
            early_stop = 0
        else:
            early_stop += 1
            if val_map > best_val_map:
                best_val_map = val_map
                torch.save(model.state_dict(), best_ce_path)
                run_summary['ce_best_val_map'] = float(best_val_map)
                run_summary['ce_best_epoch'] = epoch + 1
                run_summary['ce_checkpoint'] = best_ce_path
            print(f'No significant improvement for {early_stop}/{early_stop_patience} epochs')
            if early_stop >= early_stop_patience:
                print('Early stopping triggered (CE stage).')
                break

    if not skip_transfer and os.path.exists(best_ce_path):
        model.load_state_dict(torch.load(best_ce_path, map_location=opt.device))

    ce_test_map, ce_test_top10, ce_test_rank1 = val_slow(
        model, test_loader, opt.device, 'SHS100k Vocal Test Set (after CE)')
    print(f'CE stage test - MAP: {ce_test_map:.4f}, Top10: {ce_test_top10:.4f}, Rank1: {ce_test_rank1:.2f}')
    run_summary['ce_test_map'] = float(ce_test_map)
    run_summary['ce_test_top10'] = float(ce_test_top10)
    run_summary['ce_test_rank1'] = float(ce_test_rank1)

    with open(run_log_path, 'w') as fp:
        json.dump(run_summary, fp, indent=2)

    num_fine_tune_epochs = kwargs.get('num_fine_tune_epochs', None)

    # --- Stage 2: Triplet Loss (from CE model) ---
    print('\n=== Stage 2: Triplet Loss Fine-tuning ===')
    # Explicitly delete CE optimizer so its GPU state is freed before triplet
    del optimizer
    torch.cuda.empty_cache()
    # Reload best CE weights so triplet starts from CE model, not a partially-trained state
    model.load_state_dict(torch.load(best_ce_path, map_location=opt.device))
    fine_tune_model(
        model, train_loader, val_loader, test_loader,
        run_name, checkpoint_dir, logs_dir, run_summary, run_log_path,
    )

    # --- Stage 3: Dynamic Modified Triplet Loss (from CE model) ---
    print('\n=== Stage 3: Dynamic Modified Triplet Loss Fine-tuning ===')
    torch.cuda.empty_cache()
    # Reload best CE weights again so dynamic triplet also starts from CE model
    model.load_state_dict(torch.load(best_ce_path, map_location=opt.device))
    fine_tune_model_dynamic_triplet(
        model, train_loader, val_loader, test_loader,
        run_name, checkpoint_dir, logs_dir, run_summary, run_log_path,
        num_epochs=num_fine_tune_epochs,
        nsim_refresh_interval=kwargs.get('nsim_refresh_interval', None),
    )


def fine_tune_model(
    model, train_loader, val_loader, test_loader,
    run_name, checkpoint_dir, logs_dir, run_summary, run_log_path,
):
    num_epochs = 75
    patience = 25
    min_delta = 0.002
    best_val_map = -1.0
    early_stop = 0

    best_triplet_path = os.path.join(checkpoint_dir, f'CQTNet_shs100k_vocal_triplet_best_{run_name}.pth')
    run_summary['triplet_checkpoint'] = best_triplet_path

    for name, param in model.named_parameters():
        param.requires_grad = 'conv0' not in name

    criterion = nn.TripletMarginLoss(margin=0.05)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Smaller batch size — triplet holds 3 sets of embeddings simultaneously
    triplet_loader = DataLoader(
        train_loader.dataset,
        batch_size=64,
        shuffle=True,
        num_workers=train_loader.num_workers,
        collate_fn=custom_collate,
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        update_steps = 0

        for inputs, labels, is_orig in tqdm(triplet_loader, desc=f'Triplet Epoch {epoch + 1}/{num_epochs}'):
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)
            is_orig = is_orig.to(opt.device)

            _, embeddings = model(inputs)
            anchor, positive, negative = create_triplets(embeddings, labels, is_orig, opt.device)
            if anchor.numel() == 0:
                print('No valid triplets in this batch. Skipping.')
                continue

            loss = criterion(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            update_steps += 1

        avg_loss = total_loss / update_steps if update_steps > 0 else 0.0
        print(f'Triplet Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

        val_map, val_top10, val_rank1 = val_slow(model, val_loader, opt.device, 'SHS100k Vocal Validation')
        print(f'Validation - MAP: {val_map:.4f}, Top10: {val_top10:.4f}, Rank1: {val_rank1:.2f}')

        run_summary['triplet_history'].append({
            'epoch': epoch + 1,
            'triplet_loss': float(avg_loss),
            'val_map': float(val_map),
            'val_top10': float(val_top10),
            'val_rank1': float(val_rank1),
        })

        if val_map > best_val_map + min_delta:
            best_val_map = val_map
            torch.save(model.state_dict(), best_triplet_path)
            run_summary['triplet_best_val_map'] = float(best_val_map)
            run_summary['triplet_best_epoch'] = epoch + 1
            print(f'New best triplet model saved/updated: {best_triplet_path}')
            early_stop = 0
        else:
            early_stop += 1
            if val_map > best_val_map:
                best_val_map = val_map
                torch.save(model.state_dict(), best_triplet_path)
                run_summary['triplet_best_val_map'] = float(best_val_map)
                run_summary['triplet_best_epoch'] = epoch + 1
            print(f'No significant improvement for {early_stop}/{patience} epochs')
            if early_stop >= patience:
                print('Early stopping triggered (triplet stage).')
                break

    if os.path.exists(best_triplet_path):
        model.load_state_dict(torch.load(best_triplet_path, map_location=opt.device))

    test_map, test_top10, test_rank1 = val_slow(model, test_loader, opt.device, 'SHS100k Vocal Test Set (after Triplet)')
    print(f'Triplet stage test - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}')

    run_summary['triplet_test_map'] = float(test_map)
    run_summary['triplet_test_top10'] = float(test_top10)
    run_summary['triplet_test_rank1'] = float(test_rank1)

    with open(run_log_path, 'w') as fp:
        json.dump(run_summary, fp, indent=2)

    print(f'Run summary saved: {run_log_path}')


def fine_tune_model_modified_triplet(
    model, optimizer, train_loader, val_loader, test_loader,
    run_name, checkpoint_dir, logs_dir, run_summary, run_log_path,
    num_epochs=None,
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lookup_path = os.path.join(base_dir, 'nsim_lookup.json')
    train_txt_path = os.path.join(base_dir, 'shs100k_train.txt')
    cqt_dir = os.path.join(base_dir, 'cqt_features')

    if not os.path.exists(lookup_path):
        raise FileNotFoundError(
            f'nsim_lookup.json not found at {lookup_path}. '
            'Run compute_nsim_lookup.py first.'
        )

    nsim_map, filename_to_clique, clique_to_original = load_nsim_lookup(
        lookup_path, train_txt_path
    )
    print(f'Loaded N_sim lookup: {len(nsim_map)} originals mapped')

    label_map = train_loader.dataset.label_map

    num_epochs = num_epochs if num_epochs is not None else 75
    patience = 25
    min_delta = 0.002
    best_val_map = -1.0
    early_stop = 0
    nsim_refresh_interval = 10
    margin = 0.3

    best_path = os.path.join(
        checkpoint_dir, f'CQTNet_shs100k_vocal_mod_triplet_best_{run_name}.pth'
    )
    run_summary['mod_triplet_checkpoint'] = best_path

    # Use smaller batch size to avoid OOM (nsim embeddings use extra GPU memory)
    mod_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=120,
        shuffle=True,
        num_workers=train_loader.num_workers,
        collate_fn=custom_collate,
    )

    for name, param in model.named_parameters():
        param.requires_grad = 'conv0' not in name

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    nsim_emb_dict = compute_nsim_embeddings(
        model, nsim_map, clique_to_original, cqt_dir, label_map, opt.device
    )
    print(f'Computed N_sim embeddings for {len(nsim_emb_dict)} cliques')

    for epoch in range(num_epochs):
        if epoch > 0 and epoch % nsim_refresh_interval == 0:
            nsim_emb_dict = compute_nsim_embeddings(
                model, nsim_map, clique_to_original, cqt_dir, label_map, opt.device
            )
            print(f'Refreshed N_sim embeddings (epoch {epoch + 1})')

        model.train()
        total_loss = 0.0
        update_steps = 0

        for inputs, labels, is_orig in tqdm(
            mod_train_loader, desc=f'ModTrip Epoch {epoch + 1}/{num_epochs}'
        ):
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)
            is_orig = is_orig.to(opt.device)

            _, embeddings = model(inputs)
            anchor, positive, neg_rand, neg_sim = create_modified_triplets(
                embeddings, labels, is_orig, nsim_emb_dict, opt.device
            )
            if anchor.numel() == 0:
                continue

            loss = modified_triplet_loss(anchor, positive, neg_rand, neg_sim, margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            update_steps += 1

        avg_loss = total_loss / update_steps if update_steps > 0 else 0.0
        print(f'ModTrip Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

        val_map, val_top10, val_rank1 = val_slow(
            model, val_loader, opt.device, 'SHS100k Vocal Validation'
        )
        print(f'Validation - MAP: {val_map:.4f}, Top10: {val_top10:.4f}, Rank1: {val_rank1:.2f}')

        run_summary['mod_triplet_history'].append({
            'epoch': epoch + 1,
            'mod_triplet_loss': float(avg_loss),
            'val_map': float(val_map),
            'val_top10': float(val_top10),
            'val_rank1': float(val_rank1),
        })

        if val_map > best_val_map + min_delta:
            best_val_map = val_map
            torch.save(model.state_dict(), best_path)
            run_summary['mod_triplet_best_val_map'] = float(best_val_map)
            run_summary['mod_triplet_best_epoch'] = epoch + 1
            print(f'New best mod-triplet model saved: {best_path}')
            early_stop = 0
        else:
            early_stop += 1
            if val_map > best_val_map:
                best_val_map = val_map
                torch.save(model.state_dict(), best_path)
                run_summary['mod_triplet_best_val_map'] = float(best_val_map)
                run_summary['mod_triplet_best_epoch'] = epoch + 1
            print(f'No significant improvement for {early_stop}/{patience} epochs')
            if early_stop >= patience:
                print('Early stopping triggered (mod-triplet stage).')
                break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=opt.device))

    test_map, test_top10, test_rank1 = val_slow(
        model, test_loader, opt.device, 'SHS100k Vocal Test Set (after ModTriplet)'
    )
    print(f'ModTriplet test - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}')

    run_summary['mod_triplet_test_map'] = float(test_map)
    run_summary['mod_triplet_test_top10'] = float(test_top10)
    run_summary['mod_triplet_test_rank1'] = float(test_rank1)

    with open(run_log_path, 'w') as fp:
        json.dump(run_summary, fp, indent=2)

    print(f'Run summary saved: {run_log_path}')


@torch.no_grad()
def recompute_nsim_from_model(model, train_loader, device):
    """
    Dynamically compute N_sim using current model embeddings.
    For each clique's original, find the most similar song from a different clique
    in the learned embedding space.

    All heavy matrix operations are done on CPU to avoid GPU OOM.

    Returns:
        nsim_emb_dict: dict {label_int: embedding_tensor (D,)} stored on CPU
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_is_orig = []

    for data, labels, is_orig in tqdm(train_loader, desc='Computing embeddings for N_sim'):
        data = data.to(device)
        _, embedding = model(data)
        all_embeddings.append(embedding.detach().cpu())
        all_labels.append(labels.cpu())
        all_is_orig.append(is_orig.cpu())

    # All subsequent ops on CPU to avoid GPU OOM
    embeddings = torch.cat(all_embeddings)       # (N, D) on CPU
    labels = torch.cat(all_labels)               # (N,) on CPU
    is_orig = torch.cat(all_is_orig)             # (N,) on CPU

    # L2-normalize for cosine similarity (CPU)
    emb_norm = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = emb_norm @ emb_norm.T           # (N, N) on CPU

    nsim_emb_dict = {}

    # For each original, find the hardest negative (most similar from different clique)
    orig_indices = torch.where(is_orig)[0]
    for idx in orig_indices:
        label_val = labels[idx].item()
        sims = sim_matrix[idx].clone()

        # Mask out same-clique songs
        same_clique_mask = labels == labels[idx]
        sims[same_clique_mask] = -2.0  # below min cosine similarity

        best_neg_idx = sims.argmax().item()
        # Keep on CPU; create_modified_triplets will move to device as needed
        nsim_emb_dict[label_val] = embeddings[best_neg_idx]

    return nsim_emb_dict


def fine_tune_model_dynamic_triplet(
    model, train_loader, val_loader, test_loader,
    run_name, checkpoint_dir, logs_dir, run_summary, run_log_path,
    num_epochs=None,
    nsim_refresh_interval=None,
):
    """
    Modified triplet loss with dynamically recomputed N_sim from model embeddings.
    No dependency on nsim_lookup.json or compute_nsim_lookup.py.
    """
    num_epochs = num_epochs if num_epochs is not None else 75
    patience = 25
    min_delta = 0.002
    best_val_map = -1.0
    early_stop = 0
    nsim_refresh_interval = nsim_refresh_interval if nsim_refresh_interval is not None else 3
    margin = 0.3

    best_path = os.path.join(
        checkpoint_dir, f'CQTNet_shs100k_vocal_dyn_triplet_best_{run_name}.pth'
    )
    run_summary['dyn_triplet_history'] = []
    run_summary['dyn_triplet_best_val_map'] = None
    run_summary['dyn_triplet_best_epoch'] = None
    run_summary['dyn_triplet_checkpoint'] = best_path
    run_summary['dyn_triplet_test_map'] = None
    run_summary['dyn_triplet_test_top10'] = None
    run_summary['dyn_triplet_test_rank1'] = None

    for name, param in model.named_parameters():
        param.requires_grad = 'conv0' not in name

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5
    )

    # Reduced batch size to avoid GPU OOM during triplet construction
    dyn_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=64,
        shuffle=True,
        num_workers=train_loader.num_workers,
        collate_fn=custom_collate,
    )

    # Initial N_sim computation from model embeddings (CPU ops to avoid OOM)
    torch.cuda.empty_cache()
    nsim_emb_dict = recompute_nsim_from_model(model, dyn_train_loader, opt.device)
    print(f'Computed dynamic N_sim embeddings for {len(nsim_emb_dict)} cliques')

    for epoch in range(num_epochs):
        # Refresh N_sim embeddings periodically
        if epoch > 0 and epoch % nsim_refresh_interval == 0:
            torch.cuda.empty_cache()
            nsim_emb_dict = recompute_nsim_from_model(model, dyn_train_loader, opt.device)
            print(f'Refreshed dynamic N_sim embeddings (epoch {epoch + 1})')

        model.train()
        total_loss = 0.0
        update_steps = 0

        for inputs, labels, is_orig in tqdm(
            dyn_train_loader, desc=f'DynTrip Epoch {epoch + 1}/{num_epochs}'
        ):
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)
            is_orig = is_orig.to(opt.device)

            _, embeddings = model(inputs)
            anchor, positive, neg_rand, neg_sim = create_modified_triplets(
                embeddings, labels, is_orig, nsim_emb_dict, opt.device
            )
            if anchor.numel() == 0:
                continue

            loss = modified_triplet_loss(anchor, positive, neg_rand, neg_sim, margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            update_steps += 1

        avg_loss = total_loss / update_steps if update_steps > 0 else 0.0
        print(f'DynTrip Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

        val_map, val_top10, val_rank1 = val_slow(
            model, val_loader, opt.device, 'SHS100k Vocal Validation'
        )
        print(f'Validation - MAP: {val_map:.4f}, Top10: {val_top10:.4f}, Rank1: {val_rank1:.2f}')

        run_summary['dyn_triplet_history'].append({
            'epoch': epoch + 1,
            'dyn_triplet_loss': float(avg_loss),
            'val_map': float(val_map),
            'val_top10': float(val_top10),
            'val_rank1': float(val_rank1),
        })

        if val_map > best_val_map + min_delta:
            best_val_map = val_map
            torch.save(model.state_dict(), best_path)
            run_summary['dyn_triplet_best_val_map'] = float(best_val_map)
            run_summary['dyn_triplet_best_epoch'] = epoch + 1
            print(f'New best dynamic-triplet model saved: {best_path}')
            early_stop = 0
        else:
            early_stop += 1
            if val_map > best_val_map:
                best_val_map = val_map
                torch.save(model.state_dict(), best_path)
                run_summary['dyn_triplet_best_val_map'] = float(best_val_map)
                run_summary['dyn_triplet_best_epoch'] = epoch + 1
            print(f'No significant improvement for {early_stop}/{patience} epochs')
            if early_stop >= patience:
                print('Early stopping triggered (dynamic-triplet stage).')
                break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=opt.device))

    test_map, test_top10, test_rank1 = val_slow(
        model, test_loader, opt.device, 'SHS100k Vocal Test Set (after DynTriplet)'
    )
    print(f'DynTriplet test - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}')

    run_summary['dyn_triplet_test_map'] = float(test_map)
    run_summary['dyn_triplet_test_top10'] = float(test_top10)
    run_summary['dyn_triplet_test_rank1'] = float(test_rank1)

    with open(run_log_path, 'w') as fp:
        json.dump(run_summary, fp, indent=2)

    print(f'Run summary saved: {run_log_path}')


def evaluate(**kwargs):
    """
    Load a checkpoint and run val + test evaluation with no training.
    Usage:
        python shs100k_vocal_pipeline/main.py evaluate \
            --checkpoint=/path/to/model.pth
    """
    opt.num_workers = 4
    opt.model = 'CQTNet'
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {opt.device}')

    checkpoint = kwargs.get('checkpoint')
    if not checkpoint or not os.path.exists(checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    val_data = SHS100kVocalCQT('val')
    test_data = SHS100kVocalCQT('test')

    num_classes = SHS100kVocalCQT('train').label_count
    model = getattr(models, opt.model)()
    model.fc1 = nn.Linear(300, num_classes)
    safe_load_model(model, checkpoint)
    model = model.to(opt.device)
    torch.cuda.empty_cache()

    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=opt.num_workers, collate_fn=custom_collate)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers, collate_fn=custom_collate)

    val_map, val_top10, val_rank1 = val_slow(model, val_loader, opt.device, 'SHS100k Vocal Validation')
    test_map, test_top10, test_rank1 = val_slow(model, test_loader, opt.device, 'SHS100k Vocal Test Set')

    print('\n=== Evaluation Summary ===')
    print(f'Checkpoint : {checkpoint}')
    print(f'Val   MAP={val_map:.4f}  Top10={val_top10:.4f}  Rank1={val_rank1:.2f}')
    print(f'Test  MAP={test_map:.4f}  Top10={test_top10:.4f}  Rank1={test_rank1:.2f}')


if __name__ == '__main__':
    import fire
    fire.Fire()
