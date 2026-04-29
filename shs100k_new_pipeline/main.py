import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
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
    early_stop_patience = 25

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

        if val_map > best_val_map:
            best_val_map = val_map
            best_model_path = os.path.join(checkpoint_dir, f"CQTNet_SHS100k_transfer_learning_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
            early_stop = 0
        else:
            early_stop += 1
            if early_stop == early_stop_patience:
                break

    if best_model_path and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        
    test_map, test_top10, test_rank1 = val_slow(model, test_loader, -1, "SHS100k Test Set")
    print(f"Final Test Set Performance - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}")
    
    if kwargs.get("fine_tune") == True:
        print("\n\nFine Tuning Model with Triplet Loss....")
        fine_tune_model(model, optimizer, train_loader, val_loader, test_loader, opt, checkpoint_dir)

def fine_tune_model(model, optimizer, train_loader, val_loader, test_loader, opt, checkpoint_dir):
    num_epochs = 75
    best_val_map = 0
    best_model_path = None
    early_stop = 0
    early_stop_patience = 25
    
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
    
        if val_map > best_val_map:
            best_val_map = val_map
            best_model_path = os.path.join(checkpoint_dir, f"CQTNet_SHS100k_finetune_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
            early_stop = 0
        else:
            early_stop += 1
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
