import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import json
import random
from tqdm import tqdm
import numpy as np

# Add parent directory to path so we can import models, config, utility
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import models
from config import opt
from utility import *
from shs100k_new_pipeline.cqt_loader import SHS100kCQT

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

def transfer_learning(**kwargs):
    opt.batch_size = 120
    opt.num_workers = 2
    opt.model = 'CQTNet'
    # Start from an existing checkpoint if desired, or let it train from scratch
    opt.load_model_path = os.path.join(parent_dir, 'check_points/cqtnet_shs100k_best.pth')
    if not os.path.exists(opt.load_model_path):
        opt.load_model_path = None
        
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt._parse(kwargs)
    print(f"Using device: {opt.device}")

    # Prepare SHS100K dataset first to get the number of classes
    # We must be in PP4/ dir to match relative paths or fix paths
    train_data = SHS100kCQT('train')
    val_data = SHS100kCQT('val')
    test_data = SHS100kCQT('test')

    num_classes = train_data.set
    print(f"Number of training classes: {num_classes}")

    model = getattr(models, opt.model)()
    
    # Load base model if it exists
    if opt.load_model_path and os.path.exists(opt.load_model_path):
        try:
            model.load(opt.load_model_path)
            print(f"Loaded base model from {opt.load_model_path}")
        except Exception as e:
            print(f"Could not load base model: {e}")

    # Modify the last layer to no:of labels in our dataset
    model.fc1 = nn.Linear(300, num_classes).to(opt.device)
    model = model.to(opt.device)

    # Only 1st conv layer frozen
    for name, param in model.named_parameters():
        if 'conv0' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    skip_transfer = kwargs.get('skip_transfer', False)
    num_epochs = 100 if not skip_transfer else 0
    best_val_map = 0
    best_model_path = opt.load_model_path if skip_transfer else None
    early_stop = 0
    early_stop_patience = 15
    min_delta = 0.002

    checkpoint_dir = os.path.join(parent_dir, 'check_points')
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels, is_orig in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)

            optimizer.zero_grad()
            scores, _ = model(inputs)
            
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        val_map, val_top10, val_rank1 = val_slow(model, val_loader, epoch, "SHS100k Validation Set")
        print(f"Validation - MAP: {val_map:.4f}, Top10: {val_top10:.4f}, Rank1: {val_rank1:.2f}")

        if val_map > best_val_map + min_delta:
            best_val_map = val_map
            best_model_path = os.path.join(checkpoint_dir, f"CQTNet_SHS100k_transfer_learning_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
            early_stop = 0
        else:
            early_stop += 1
            if val_map > best_val_map:
                best_val_map = val_map
                best_model_path = os.path.join(checkpoint_dir, f"CQTNet_SHS100k_transfer_learning_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), best_model_path)
            print(f"No significant improvement for {early_stop}/{early_stop_patience} epochs")
            if early_stop == early_stop_patience:
                break

    if best_model_path and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        
    test_map, test_top10, test_rank1 = val_slow(model, test_loader, -1, "SHS100k Test Set")
    print(f"Final Test Set Performance - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}")
    
    if kwargs.get("modified_triplet") == True:
        print("\n\nFine Tuning Model with Modified Triplet Loss....")
        fine_tune_model_modified_triplet(model, optimizer, train_loader, val_loader, test_loader, opt, checkpoint_dir)
    elif kwargs.get("fine_tune") == True:
        print("\n\nFine Tuning Model with Triplet Loss....")
        fine_tune_model(model, optimizer, train_loader, val_loader, test_loader, opt, checkpoint_dir)

def fine_tune_model(model, optimizer, train_loader, val_loader, test_loader, opt, checkpoint_dir):
    num_epochs = 75
    best_val_map = 0
    best_model_path = None
    early_stop = 0
    early_stop_patience = 15
    min_delta = 0.002
    
    for name, param in model.named_parameters():
        if 'conv0' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    criterion = nn.TripletMarginLoss(margin=0.05)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels, is_orig in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels, is_orig = inputs.to(opt.device), labels.to(opt.device), is_orig.to(opt.device)
            _, embeddings = model(inputs)
            # Create triplets
            anchor, positive, negative = create_triplets(embeddings, labels, is_orig)
            if anchor.size(0) > 0:  # Check if we have valid triplets
                loss = criterion(anchor, positive, negative)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                print("No valid triplets in this batch. Skipping.")
        
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
        # Evaluate on validation set
        val_map, val_top10, val_rank1 = val_slow(model, val_loader, epoch, "SHS100k Validation Set")
        print(f"Validation - MAP: {val_map:.4f}, Top10: {val_top10:.4f}, Rank1: {val_rank1:.2f}")
    
        if val_map > best_val_map + min_delta:
            best_val_map = val_map
            best_model_path = os.path.join(checkpoint_dir, f"CQTNet_SHS100k_finetune_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
            early_stop = 0
        else:
            early_stop += 1
            if val_map > best_val_map:
                best_val_map = val_map
                best_model_path = os.path.join(checkpoint_dir, f"CQTNet_SHS100k_finetune_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), best_model_path)
            print(f"No significant improvement for {early_stop}/{early_stop_patience} epochs")
            if early_stop == early_stop_patience:
                break
    
    if best_model_path and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    test_map, test_top10, test_rank1 = val_slow(model, test_loader, -1, "SHS100k Test Set")
    print(f"Final Test Set Performance - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}")

def create_triplets(embeddings, labels, is_orig):
    anchors = []
    positives = []
    negatives = []
    
    for i in range(len(embeddings)):
        anchor = embeddings[i].unsqueeze(0)
        anchor_label = labels[i]
        
        pos_mask = labels == anchor_label
        pos_mask[i] = False
        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(labels != anchor_label)[0]
        
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue
            
        pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,)).item()]
        neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,)).item()]
        
        anchors.append(anchor)
        positives.append(embeddings[pos_idx].unsqueeze(0))
        negatives.append(embeddings[neg_idx].unsqueeze(0))
            
    if len(anchors) == 0:
        return torch.tensor([]).to(opt.device), torch.tensor([]).to(opt.device), torch.tensor([]).to(opt.device)

    return torch.cat(anchors).to(opt.device), torch.cat(positives).to(opt.device), torch.cat(negatives).to(opt.device)


def modified_triplet_loss(anchor, positive, neg_rand, neg_sim, margin=0.3):
    d_pos = 1 - F.cosine_similarity(anchor, positive, dim=-1)
    d_rand = 1 - F.cosine_similarity(anchor, neg_rand, dim=-1)
    d_sim = 1 - F.cosine_similarity(anchor, neg_sim, dim=-1)
    loss = torch.clamp(margin + 2 * d_pos - (d_sim + d_rand), min=0.0)
    return loss.mean()


def create_modified_triplets(embeddings, labels, is_orig, nsim_emb_dict):
    anchors = []
    positives = []
    neg_rands = []
    neg_sims = []

    for i in range(len(embeddings)):
        anchor_label = labels[i].item()
        pos_mask = labels == labels[i]
        pos_mask[i] = False
        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(labels != labels[i])[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue

        pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,)).item()]
        neg_r_idx = neg_indices[torch.randint(0, len(neg_indices), (1,)).item()]

        anchors.append(embeddings[i].unsqueeze(0))
        positives.append(embeddings[pos_idx].unsqueeze(0))
        neg_rands.append(embeddings[neg_r_idx].unsqueeze(0))

        if anchor_label in nsim_emb_dict:
            neg_sims.append(nsim_emb_dict[anchor_label].unsqueeze(0))
        else:
            neg_sims.append(embeddings[neg_r_idx].unsqueeze(0))

    if len(anchors) == 0:
        empty = torch.tensor([]).to(opt.device)
        return empty, empty, empty, empty

    return (
        torch.cat(anchors).to(opt.device),
        torch.cat(positives).to(opt.device),
        torch.cat(neg_rands).to(opt.device),
        torch.cat(neg_sims).to(opt.device),
    )


def load_nsim_lookup(lookup_path, train_txt_path):
    with open(lookup_path) as f:
        lookup = json.load(f)

    filename_to_clique = {}
    clique_to_original = {}
    with open(train_txt_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if not parts[0]:
                continue
            filename_to_clique[parts[0]] = parts[1]
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
                nsim_emb_dict[label_map[clique]] = emb.squeeze(0)
    return nsim_emb_dict


def fine_tune_model_modified_triplet(model, optimizer, train_loader, val_loader, test_loader, opt, checkpoint_dir):
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

    label_map = train_loader.dataset.set_dict

    num_epochs = 75
    best_val_map = 0
    best_model_path = None
    early_stop = 0
    early_stop_patience = 15
    min_delta = 0.002
    nsim_refresh_interval = 10
    margin = 0.3

    for name, param in model.named_parameters():
        if 'conv0' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    nsim_emb_dict = compute_nsim_embeddings(
        model, nsim_map, clique_to_original, cqt_dir, label_map, opt.device
    )
    print(f'Computed N_sim embeddings for {len(nsim_emb_dict)} cliques')

    for epoch in range(num_epochs):
        if epoch > 0 and epoch % nsim_refresh_interval == 0:
            nsim_emb_dict = compute_nsim_embeddings(
                model, nsim_map, clique_to_original, cqt_dir, label_map, opt.device
            )
            print(f'Refreshed N_sim embeddings (epoch {epoch+1})')

        model.train()
        total_loss = 0
        update_steps = 0
        for inputs, labels, is_orig in tqdm(train_loader, desc=f"ModTrip Epoch {epoch+1}/{num_epochs}"):
            inputs, labels, is_orig = inputs.to(opt.device), labels.to(opt.device), is_orig.to(opt.device)
            _, embeddings = model(inputs)
            anchor, positive, neg_rand, neg_sim = create_modified_triplets(
                embeddings, labels, is_orig, nsim_emb_dict
            )
            if anchor.numel() == 0:
                continue

            loss = modified_triplet_loss(anchor, positive, neg_rand, neg_sim, margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            update_steps += 1

        avg_loss = total_loss / update_steps if update_steps > 0 else 0
        print(f"ModTrip Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        val_map, val_top10, val_rank1 = val_slow(model, val_loader, epoch, "SHS100k Validation Set")
        print(f"Validation - MAP: {val_map:.4f}, Top10: {val_top10:.4f}, Rank1: {val_rank1:.2f}")

        if val_map > best_val_map + min_delta:
            best_val_map = val_map
            best_model_path = os.path.join(checkpoint_dir, f"CQTNet_SHS100k_mod_triplet_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
            early_stop = 0
        else:
            early_stop += 1
            if val_map > best_val_map:
                best_val_map = val_map
                best_model_path = os.path.join(checkpoint_dir, f"CQTNet_SHS100k_mod_triplet_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), best_model_path)
            print(f"No significant improvement for {early_stop}/{early_stop_patience} epochs")
            if early_stop == early_stop_patience:
                break

    if best_model_path and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    test_map, test_top10, test_rank1 = val_slow(model, test_loader, -1, "SHS100k Test Set")
    print(f"Final Test Set Performance - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}")


@torch.no_grad()
def val_slow(model, dataloader, epoch, dataset_name=None):
    model.eval()
    all_embeddings = []
    all_labels = []

    for data, label, _ in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        input = data.to(opt.device)
        _, embedding = model(input)
        all_embeddings.append(embedding.cpu().numpy())
        all_labels.append(label.cpu().numpy())

    if not all_embeddings:
        return 0, 0, 0

    embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_labels)
    
    embeddings = norm(embeddings)
    dis2d = -np.matmul(embeddings, embeddings.T)
    
    MAP, top10, rank1 = calc_MAP(dis2d, labels)

    print(f"\nResults for {dataset_name}:")
    print(f"MAP: {MAP:.4f}, Top10: {top10:.4f}, Rank1: {rank1:.2f}")
    return MAP, top10, rank1

if __name__=='__main__':
    import fire
    fire.Fire()
