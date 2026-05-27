import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import models
from config import opt
from utility import *
from tqdm import tqdm
import numpy as np
import json
from cqt_loader import IndianCoverCQT
import random

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
    opt.batch_size = 60
    opt.num_workers = 2
    opt.model = 'CQTNet'
    opt.load_model_path = 'check_points/cqtnet_indian_best.pth'
    # opt.load_model_path = '/content/CQTNet/check_points/latest.pth'
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt._parse(kwargs)
    print(f"Using device: {opt.device}")

    # Load pre-trained model
    model = getattr(models, opt.model)()
    # Modify the last layer to no:of labels in our dataset before loading
    model.fc1 = nn.Linear(300, 402).to(opt.device)
    model.load(opt.load_model_path)
    model = model.to(opt.device)

    # Only 1st conv layer frozen
    for name, param in model.named_parameters():
       if 'conv0' in name:
           param.requires_grad = False
       else:
           param.requires_grad = True
    """
    # Freeze all layers except the last two
    for name, param in model.named_parameters():
        if 'fc0' not in name and 'fc1' not in name:
            param.requires_grad = False
    """

    # Prepare Indian Cover Songs dataset
    train_data = IndianCoverCQT('train')
    val_data = IndianCoverCQT('val')
    test_data = IndianCoverCQT('test')

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=custom_collate)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Training loop
    skip_transfer = kwargs.get('skip_transfer', False)
    num_epochs = 200 if not skip_transfer else 0
    best_val_map = 0
    best_model_path = opt.load_model_path if skip_transfer else None
    early_stop = 0
    early_stop_patience = 25

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
        val_map, val_top10, val_rank1 = val_slow(model, val_loader, epoch, "Indian Validation Set")
        print(f"Validation - MAP: {val_map:.4f}, Top10: {val_top10:.4f}, Rank1: {val_rank1:.2f}")

        if val_map > best_val_map:
            best_val_map = val_map
            best_model_path = f"check_points/CQTNet_transfer_learning_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
            early_stop = 0
        else:
            early_stop+=1
            if early_stop == early_stop_patience:
                break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path))
    torch.cuda.empty_cache()
    test_map, test_top10, test_rank1 = val_slow(model, test_loader, -1, "Indian Test Set")
    print(f"Final Test Set Performance - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}")
    
    if kwargs.get("modified_triplet")==True:
        print("\n\nFine Tuning Model with Modified Triplet Loss....")
        fine_tune_model_modified_triplet(model, optimizer, train_loader, val_loader, test_loader, opt)
    elif kwargs.get("fine_tune")==True:
        print("\n\nFine Tuning Model with Triplet Loss....")
        fine_tune_model(model, optimizer, train_loader, val_loader, test_loader, opt)

def fine_tune_model(model, optimizer, train_loader, val_loader, test_loader, opt):
    num_epochs = 150
    best_val_map = 0
    best_model_path = None
    early_stop = 0
    early_stop_patience = 25

    #Convert to Embedding Layer
    #model.fc1 = nn.Linear(300, 300).to(opt.device)
    
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
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
        # Evaluate on validation set
        val_map, val_top10, val_rank1 = val_slow(model, val_loader, epoch, "Indian Validation Set")
        print(f"Validation - MAP: {val_map:.4f}, Top10: {val_top10:.4f}, Rank1: {val_rank1:.2f}")
    
        if val_map > best_val_map:
            best_val_map = val_map
            best_model_path = f"check_points/CQTNet_transfer_learning_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
            early_stop = 0
        else:
            early_stop+=1
            if early_stop == early_stop_patience:
                break
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path))
    torch.cuda.empty_cache()
    test_map, test_top10, test_rank1 = val_slow(model, test_loader, -1, "Indian Test Set")
    print(f"Final Test Set Performance - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}")

def fine_tune_model_modified_triplet(model, optimizer, train_loader, val_loader, test_loader, opt):
    lookup_path = 'indian_new_pipeline/nsim_lookup.json'
    train_txt_path = 'indian_new_pipeline/indian_train.txt'
    cqt_dir = 'indian_new_pipeline/cqt_features'

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

    num_epochs = 150
    best_val_map = 0
    best_model_path = None
    early_stop = 0
    early_stop_patience = 25
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

        val_map, val_top10, val_rank1 = val_slow(model, val_loader, epoch, "Indian Validation Set")
        print(f"Validation - MAP: {val_map:.4f}, Top10: {val_top10:.4f}, Rank1: {val_rank1:.2f}")

        if val_map > best_val_map:
            best_val_map = val_map
            best_model_path = f"check_points/CQTNet_mod_triplet_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
            early_stop = 0
        else:
            early_stop += 1
            if early_stop == early_stop_patience:
                break

    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
    test_map, test_top10, test_rank1 = val_slow(model, test_loader, -1, "Indian Test Set")
    print(f"Final Test Set Performance - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}")

def load(in_path):
    data = np.load(in_path)
    transform_test = transforms.Compose([
        lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
        lambda x: torch.Tensor(x),
        lambda x: x.unsqueeze(0),  # Add channel dimension
    ])
    data = transform_test(data)
    data = pad_or_truncate(data, 400, 84)
    return data.unsqueeze(0).to(opt.device)

# Including necessary methods
def pad_or_truncate(data, target_length, target_freq=84):
    if data.ndim == 2:
        data = data.unsqueeze(0)
    _, freq, current_length = data.shape
    if freq > target_freq:
        data = data[:, :target_freq, :]
    elif freq < target_freq:
        pad_freq = target_freq - freq
        data = F.pad(data, (0, 0, 0, pad_freq), mode='constant', value=0)
    if current_length > target_length:
        data = data[:, :, :target_length]
    elif current_length < target_length:
        pad_time = target_length - current_length
        data = F.pad(data, (0, pad_time), mode='constant', value=0)
    return data

def create_triplets(embeddings, labels, is_orig):
    """
    Create triplets for triplet loss.
    For each original track (anchor), select:
      - a random positive (cover)
      - a random negative
    """
    anchors = []
    positives = []
    negatives = []
    
    for i in range(len(embeddings)):
        if is_orig[i]:
            anchor = embeddings[i].unsqueeze(0)
            anchor_label = labels[i]
            
            pos_indices = torch.where((labels == anchor_label) & (~is_orig))[0]
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
    """
    Modified triplet loss from the paper (Section 1.6).
    L = max(0, margin + 2·d(A, P) - [d(A, N_sim) + d(A, N_rand)])
    where d(·) = 1 - cosine_similarity.
    """
    d_pos = 1 - F.cosine_similarity(anchor, positive, dim=-1)
    d_rand = 1 - F.cosine_similarity(anchor, neg_rand, dim=-1)
    d_sim = 1 - F.cosine_similarity(anchor, neg_sim, dim=-1)
    loss = torch.clamp(margin + 2 * d_pos - (d_sim + d_rand), min=0.0)
    return loss.mean()


def create_modified_triplets(embeddings, labels, is_orig, nsim_emb_dict):
    """
    Create 4-tuples for modified triplet loss.
    """
    anchors = []
    positives = []
    neg_rands = []
    neg_sims = []

    for i in range(len(embeddings)):
        if is_orig[i]:
            anchor_label = labels[i].item()
            pos_indices = torch.where((labels == labels[i]) & (~is_orig))[0]
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
    with open(train_txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            filename_to_clique[parts[0]] = parts[1]

    clique_to_original = {}
    for fname, clique in filename_to_clique.items():
        if fname.endswith('-o') or '-o-' in fname:
            clique_to_original[clique] = fname

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
            data = load(npy_path)
            _, emb = model(data)
            if clique in label_map:
                nsim_emb_dict[label_map[clique]] = emb.squeeze(0)
    return nsim_emb_dict


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
