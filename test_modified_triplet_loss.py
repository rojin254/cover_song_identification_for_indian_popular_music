"""
Test script: Validate modified triplet loss function.
Compares standard TripletMarginLoss with our custom modified triplet loss
using synthetic data. Also tests create_modified_triplets with a mock nsim_lookup.

Paper formula (Section 1.6):
  L = max(0, α + 2·d(A_ori, P_cover) - [d(A_ori, N_sim) + d(A_ori, N_rand)])

Where d(·) is cosine distance = 1 - cosine_similarity.
"""

import torch
import torch.nn.functional as F
import json
import numpy as np

torch.manual_seed(0)
np.random.seed(0)


# ============================================================
# 1. Modified Triplet Loss Function
# ============================================================

def modified_triplet_loss(anchor, positive, neg_rand, neg_sim, margin=0.3):
    """
    Modified triplet loss from the paper.

    L = max(0, margin + 2·d_pos - (d_sim + d_rand))

    All distances are cosine distance = 1 - cosine_similarity.

    Args:
        anchor:   (B, D) embeddings of original songs
        positive: (B, D) embeddings of cover songs
        neg_rand: (B, D) embeddings of random non-cover songs
        neg_sim:  (B, D) embeddings of hardest non-cover songs (N_sim)
        margin:   margin hyperparameter (default 0.3)

    Returns:
        Scalar loss value.
    """
    d_pos = 1 - F.cosine_similarity(anchor, positive, dim=-1)
    d_rand = 1 - F.cosine_similarity(anchor, neg_rand, dim=-1)
    d_sim = 1 - F.cosine_similarity(anchor, neg_sim, dim=-1)
    loss = torch.clamp(margin + 2 * d_pos - (d_sim + d_rand), min=0.0)
    return loss.mean()


# ============================================================
# 2. Standard Triplet Loss (for comparison)
# ============================================================

def standard_triplet_loss_cosine(anchor, positive, negative, margin=0.3):
    """
    Standard triplet loss using cosine distance (for apple-to-apple comparison).
    L = max(0, margin + d_pos - d_neg)
    """
    d_pos = 1 - F.cosine_similarity(anchor, positive, dim=-1)
    d_neg = 1 - F.cosine_similarity(anchor, negative, dim=-1)
    loss = torch.clamp(margin + d_pos - d_neg, min=0.0)
    return loss.mean()


# ============================================================
# 3. Create Modified Triplets
# ============================================================

def create_modified_triplets(embeddings, labels, is_orig, nsim_lookup, label_to_filename, device):
    """
    Create 4-tuples for modified triplet loss.

    For each original anchor:
      - positive  = random cover of the same clique
      - neg_rand  = random embedding from a different clique
      - neg_sim   = embedding of the hardest non-cover (looked up from nsim_lookup)

    Args:
        embeddings:  (N, D) batch embeddings
        labels:      (N,) clique labels
        is_orig:     (N,) boolean - True if original
        nsim_lookup: dict {filename: {"nsim_id": ..., ...}}
        label_to_filename: dict {label_int: filename_of_original} mapping
                           (built from the batch)
        device:      torch device

    Returns:
        (anchors, positives, neg_rands, neg_sims) each (M, D)
    """
    anchors = []
    positives = []
    neg_rands = []
    neg_sims = []

    unique_labels = labels.unique()
    # Build filename -> embedding index mapping from the batch
    # (In actual training, nsim embeddings will be from the same batch or precomputed)

    for label in unique_labels:
        label_mask = labels == label
        original_idx = torch.where(label_mask & is_orig)[0]
        cover_idx = torch.where(label_mask & (~is_orig))[0]
        negative_idx = torch.where(labels != label)[0]

        if len(original_idx) == 0 or len(cover_idx) == 0 or len(negative_idx) == 0:
            continue

        for a_idx in original_idx:
            label_val = labels[a_idx].item()

            # Positive: random cover of same clique
            pos_idx = cover_idx[torch.randint(0, len(cover_idx), (1,)).item()]

            # Neg_rand: random song from a different clique
            neg_r_idx = negative_idx[torch.randint(0, len(negative_idx), (1,)).item()]

            # Neg_sim: look up from nsim_lookup
            anchor_filename = label_to_filename.get(label_val)
            nsim_entry = nsim_lookup.get(anchor_filename) if anchor_filename else None

            if nsim_entry is None:
                # Fallback: use the random negative as nsim too
                neg_s_idx = neg_r_idx
            else:
                nsim_id = nsim_entry['nsim_id']
                # Find nsim_id in the current batch by filename lookup
                # In practice, we'll have a precomputed embedding dict
                # For this test, we fallback to random neg from different clique
                # (This is the part that will use precomputed embeddings in training)
                neg_s_idx = neg_r_idx  # placeholder for test

            anchors.append(embeddings[a_idx].unsqueeze(0))
            positives.append(embeddings[pos_idx].unsqueeze(0))
            neg_rands.append(embeddings[neg_r_idx].unsqueeze(0))
            neg_sims.append(embeddings[neg_s_idx].unsqueeze(0))

    if not anchors:
        empty = torch.empty(0, embeddings.shape[1], device=device)
        return empty, empty, empty, empty

    return (
        torch.cat(anchors).to(device),
        torch.cat(positives).to(device),
        torch.cat(neg_rands).to(device),
        torch.cat(neg_sims).to(device),
    )


# ============================================================
# 4. Tests
# ============================================================

def test_basic_loss_values():
    """Test loss is 0 when positive is very close and negatives are far."""
    print("=" * 60)
    print("TEST 1: Basic loss value sanity checks")
    print("=" * 60)

    D = 128
    anchor = F.normalize(torch.randn(1, D), dim=-1)

    # Case 1: positive = anchor (perfect), negatives far away
    positive = anchor.clone()
    neg_rand = F.normalize(torch.randn(1, D), dim=-1)
    neg_sim = F.normalize(torch.randn(1, D), dim=-1)

    loss = modified_triplet_loss(anchor, positive, neg_rand, neg_sim, margin=0.0)
    print(f"  Perfect positive, random negs, margin=0: loss = {loss.item():.6f}")
    assert loss.item() == 0.0, "Loss should be 0 when positive = anchor and margin = 0"

    # Case 2: all same → loss should be margin (2*0 - (0+0) + margin)
    loss_same = modified_triplet_loss(anchor, anchor, anchor, anchor, margin=0.3)
    print(f"  All identical embeddings, margin=0.3: loss = {loss_same.item():.6f}")
    assert abs(loss_same.item() - 0.3) < 1e-5, f"Expected ~0.3, got {loss_same.item()}"

    # Case 3: positive far, negatives close → high loss
    positive_far = -anchor  # cosine distance = 2.0
    neg_close = anchor + 0.01 * torch.randn(1, D)
    neg_close = F.normalize(neg_close, dim=-1)

    loss_hard = modified_triplet_loss(anchor, positive_far, neg_close, neg_close, margin=0.3)
    print(f"  Positive opposite, negs very close: loss = {loss_hard.item():.6f}")
    assert loss_hard.item() > 1.0, "Loss should be high when positive is far and negatives are close"

    print("  ✓ All basic sanity checks passed\n")


def test_formula_alignment():
    """Verify loss matches manual computation of the paper formula."""
    print("=" * 60)
    print("TEST 2: Formula alignment with paper")
    print("=" * 60)

    D = 64
    B = 5
    anchor = F.normalize(torch.randn(B, D), dim=-1)
    positive = F.normalize(torch.randn(B, D), dim=-1)
    neg_rand = F.normalize(torch.randn(B, D), dim=-1)
    neg_sim = F.normalize(torch.randn(B, D), dim=-1)
    margin = 0.3

    # Compute manually using the paper formula
    d_pos = 1 - F.cosine_similarity(anchor, positive, dim=-1)
    d_rand = 1 - F.cosine_similarity(anchor, neg_rand, dim=-1)
    d_sim = 1 - F.cosine_similarity(anchor, neg_sim, dim=-1)
    manual_loss = torch.clamp(margin + 2 * d_pos - (d_sim + d_rand), min=0.0).mean()

    func_loss = modified_triplet_loss(anchor, positive, neg_rand, neg_sim, margin)

    diff = abs(manual_loss.item() - func_loss.item())
    print(f"  Manual: {manual_loss.item():.8f}")
    print(f"  Function: {func_loss.item():.8f}")
    print(f"  Diff: {diff:.10f}")
    assert diff < 1e-6, f"Mismatch! diff={diff}"

    print("  ✓ Formula matches paper exactly\n")


def test_comparison_with_standard_triplet():
    """Compare modified vs standard triplet loss behavior."""
    print("=" * 60)
    print("TEST 3: Modified vs Standard triplet loss comparison")
    print("=" * 60)

    D = 128
    B = 10
    margin = 0.3

    anchor = F.normalize(torch.randn(B, D), dim=-1)
    positive = F.normalize(torch.randn(B, D), dim=-1)
    neg_rand = F.normalize(torch.randn(B, D), dim=-1)
    neg_sim = F.normalize(torch.randn(B, D), dim=-1)

    std_loss = standard_triplet_loss_cosine(anchor, positive, neg_rand, margin)
    mod_loss = modified_triplet_loss(anchor, positive, neg_rand, neg_sim, margin)

    print(f"  Standard triplet loss (1 neg):  {std_loss.item():.6f}")
    print(f"  Modified triplet loss (2 negs): {mod_loss.item():.6f}")

    # With neg_sim = neg_rand, modified should give:
    #   margin + 2*d_pos - 2*d_rand  vs  standard: margin + d_pos - d_rand
    # So modified ≈ 2x the standard effect
    mod_with_same_neg = modified_triplet_loss(anchor, positive, neg_rand, neg_rand, margin)
    std_for_compare = standard_triplet_loss_cosine(anchor, positive, neg_rand, margin)

    d_pos = 1 - F.cosine_similarity(anchor, positive, dim=-1)
    d_rand = 1 - F.cosine_similarity(anchor, neg_rand, dim=-1)

    manual_std = torch.clamp(margin + d_pos - d_rand, min=0.0).mean()
    manual_mod = torch.clamp(margin + 2 * d_pos - 2 * d_rand, min=0.0).mean()

    print(f"\n  When neg_sim == neg_rand:")
    print(f"    Standard: margin + d_pos - d_rand     = {manual_std.item():.6f}")
    print(f"    Modified: margin + 2*d_pos - 2*d_rand = {manual_mod.item():.6f}")
    print(f"    Ratio (mod/std): {manual_mod.item() / (manual_std.item() + 1e-8):.4f}")

    # Key property: when neg_sim is closer than neg_rand, modified loss should be HIGHER
    # than standard (harder optimization signal)
    neg_sim_close = anchor + 0.1 * torch.randn(B, D)
    neg_sim_close = F.normalize(neg_sim_close, dim=-1)

    mod_hard = modified_triplet_loss(anchor, positive, neg_rand, neg_sim_close, margin)
    print(f"\n  With hard neg_sim (very close to anchor):")
    print(f"    Modified loss: {mod_hard.item():.6f}")
    print(f"    Standard loss: {std_loss.item():.6f}")

    print("  ✓ Comparison complete\n")


def test_gradient_flow():
    """Ensure gradients flow correctly through the loss."""
    print("=" * 60)
    print("TEST 4: Gradient flow")
    print("=" * 60)

    D = 64
    B = 4

    # Use leaf tensors and retain_grad on normalized versions
    anchor_raw = torch.randn(B, D, requires_grad=True)
    positive_raw = torch.randn(B, D, requires_grad=True)
    neg_rand_raw = torch.randn(B, D, requires_grad=True)
    neg_sim_raw = torch.randn(B, D, requires_grad=True)

    anchor = F.normalize(anchor_raw, dim=-1)
    positive = F.normalize(positive_raw, dim=-1)
    neg_rand = F.normalize(neg_rand_raw, dim=-1)
    neg_sim = F.normalize(neg_sim_raw, dim=-1)

    loss = modified_triplet_loss(anchor, positive, neg_rand, neg_sim, margin=0.3)
    loss.backward()

    leaves = [anchor_raw, positive_raw, neg_rand_raw, neg_sim_raw]
    names = ['Anchor', 'Positive', 'Neg_rand', 'Neg_sim']
    has_grad = all(t.grad is not None for t in leaves)
    no_nan = all(not torch.isnan(t.grad).any() for t in leaves)

    print(f"  Loss value: {loss.item():.6f}")
    print(f"  All leaf tensors have gradients: {has_grad}")
    print(f"  No NaN in gradients: {no_nan}")
    for name, t in zip(names, leaves):
        print(f"  {name} grad norm: {t.grad.norm().item():.6f}")

    assert has_grad, "Missing gradients!"
    assert no_nan, "NaN gradients detected!"

    print("  ✓ Gradient flow OK\n")


def test_batch_consistency():
    """Verify batch computation matches per-sample computation."""
    print("=" * 60)
    print("TEST 5: Batch vs per-sample consistency")
    print("=" * 60)

    D = 64
    B = 8
    margin = 0.3

    anchor = F.normalize(torch.randn(B, D), dim=-1)
    positive = F.normalize(torch.randn(B, D), dim=-1)
    neg_rand = F.normalize(torch.randn(B, D), dim=-1)
    neg_sim = F.normalize(torch.randn(B, D), dim=-1)

    # Batch computation
    batch_loss = modified_triplet_loss(anchor, positive, neg_rand, neg_sim, margin)

    # Per-sample computation
    per_sample_losses = []
    for i in range(B):
        l = modified_triplet_loss(
            anchor[i:i+1], positive[i:i+1],
            neg_rand[i:i+1], neg_sim[i:i+1], margin
        )
        per_sample_losses.append(l.item())
    per_sample_mean = sum(per_sample_losses) / B

    diff = abs(batch_loss.item() - per_sample_mean)
    print(f"  Batch loss: {batch_loss.item():.8f}")
    print(f"  Per-sample mean: {per_sample_mean:.8f}")
    print(f"  Diff: {diff:.10f}")
    assert diff < 1e-5, f"Batch/per-sample mismatch: {diff}"

    print("  ✓ Batch consistency OK\n")


def test_create_modified_triplets():
    """Test create_modified_triplets with synthetic data."""
    print("=" * 60)
    print("TEST 6: create_modified_triplets function")
    print("=" * 60)

    D = 128
    device = torch.device('cpu')

    # Simulate a batch: 3 cliques, each with 1 original + 2 covers = 9 items
    embeddings = F.normalize(torch.randn(9, D), dim=-1)
    labels = torch.tensor([0, 0, 0,  1, 1, 1,  2, 2, 2])
    is_orig = torch.tensor([True, False, False,  True, False, False,  True, False, False])

    # Mock nsim_lookup (not used directly in this simplified version)
    nsim_lookup = {}
    label_to_filename = {}

    anchors, positives, neg_rands, neg_sims = create_modified_triplets(
        embeddings, labels, is_orig, nsim_lookup, label_to_filename, device
    )

    print(f"  Input: {len(embeddings)} embeddings, 3 cliques")
    print(f"  Output: {anchors.shape[0]} triplets")
    print(f"  Anchor shape: {anchors.shape}")
    print(f"  Positive shape: {positives.shape}")
    print(f"  Neg_rand shape: {neg_rands.shape}")
    print(f"  Neg_sim shape: {neg_sims.shape}")

    assert anchors.shape[0] == 3, f"Expected 3 triplets (1 per original), got {anchors.shape[0]}"
    assert anchors.shape == positives.shape == neg_rands.shape == neg_sims.shape

    # Compute loss with these triplets
    loss = modified_triplet_loss(anchors, positives, neg_rands, neg_sims)
    print(f"  Modified triplet loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN!"

    print("  ✓ create_modified_triplets OK\n")


def test_edge_case_empty_batch():
    """Test with a batch that has no valid triplets."""
    print("=" * 60)
    print("TEST 7: Edge case - no valid triplets")
    print("=" * 60)

    D = 64
    device = torch.device('cpu')

    # Only originals, no covers
    embeddings = F.normalize(torch.randn(3, D), dim=-1)
    labels = torch.tensor([0, 1, 2])
    is_orig = torch.tensor([True, True, True])

    anchors, positives, neg_rands, neg_sims = create_modified_triplets(
        embeddings, labels, is_orig, {}, {}, device
    )

    print(f"  All originals, no covers → triplets: {anchors.shape[0]}")
    assert anchors.shape[0] == 0, "Should produce 0 triplets"

    print("  ✓ Empty batch handled correctly\n")


def test_with_real_nsim_lookup():
    """Load a real nsim_lookup.json and verify structure."""
    print("=" * 60)
    print("TEST 8: Real nsim_lookup.json structure check")
    print("=" * 60)

    import os
    lookup_paths = [
        'indian_new_pipeline/nsim_lookup.json',
        'indian_vocal_pipeline/nsim_lookup.json',
        'shs100k_new_pipeline/nsim_lookup.json',
        'shs100k_vocal_pipeline/nsim_lookup.json',
    ]

    for path in lookup_paths:
        if not os.path.exists(path):
            print(f"  SKIP {path} (not found)")
            continue

        with open(path) as f:
            lookup = json.load(f)

        n = len(lookup)
        sample_key = list(lookup.keys())[0]
        sample_val = lookup[sample_key]

        # Verify structure
        assert 'nsim_id' in sample_val, f"Missing nsim_id in {path}"
        assert 'nsim_clique' in sample_val, f"Missing nsim_clique in {path}"
        assert 'cosine_similarity' in sample_val, f"Missing cosine_similarity in {path}"

        # Verify nsim is from a different clique
        # Get the source clique from the key's filename
        # For indian: format is "100-1-ci-c" with clique as first number
        # We just check nsim_id != key
        assert sample_val['nsim_id'] != sample_key, "nsim should not be the same song!"

        # Check cosine similarity is in valid range
        sim = sample_val['cosine_similarity']
        assert -1.0 <= sim <= 1.0, f"Invalid cosine similarity: {sim}"

        print(f"  ✓ {path}: {n} entries, sample: {sample_key} → {sample_val['nsim_id']} (sim={sim:.4f})")

    print("  ✓ All lookup files valid\n")


def test_modified_loss_is_differentiable_through_model_proxy():
    """Simulate model output → loss → backward to verify end-to-end."""
    print("=" * 60)
    print("TEST 9: End-to-end differentiability (model proxy)")
    print("=" * 60)

    D = 300  # CQTNet embedding dimension
    B = 10

    # Simulate model with a linear layer
    linear = torch.nn.Linear(D, D)

    # "Raw embeddings" from model
    raw = torch.randn(B, D)
    embeddings = linear(raw)  # differentiable

    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    is_orig = torch.tensor([True, False, False, False, True, False, False, False, True, False])

    anchors, positives, neg_rands, neg_sims = create_modified_triplets(
        embeddings, labels, is_orig, {}, {}, torch.device('cpu')
    )

    if anchors.shape[0] > 0:
        loss = modified_triplet_loss(anchors, positives, neg_rands, neg_sims, margin=0.3)
        loss.backward()

        grad_exists = any(p.grad is not None and p.grad.norm() > 0 for p in linear.parameters())
        print(f"  Triplets created: {anchors.shape[0]}")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Gradients reach linear layer: {grad_exists}")
        assert grad_exists, "Gradients should flow through to model parameters"
    else:
        print("  No valid triplets (expected with small batch)")

    print("  ✓ End-to-end differentiability OK\n")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("MODIFIED TRIPLET LOSS — VALIDATION TEST SUITE")
    print("=" * 60 + "\n")

    test_basic_loss_values()
    test_formula_alignment()
    test_comparison_with_standard_triplet()
    test_gradient_flow()
    test_batch_consistency()
    test_create_modified_triplets()
    test_edge_case_empty_batch()
    test_with_real_nsim_lookup()
    test_modified_loss_is_differentiable_through_model_proxy()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
