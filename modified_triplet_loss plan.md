# Plan: N_sim Generation + Modified Triplet Loss

## Addressing the Anchor/Cover Doubt

The paper's modified triplet loss formula is always computed from the **original as anchor** (`A_ori`). All four components — anchor, positive, N_rand, N_sim — are measured as distances *from the original*. This means:

- **N_sim is a property of the original anchor**, not the cover.
- When training a triplet `(original, cover_positive, N_rand, N_sim)`, the N_sim is the augmented version of the **original** (the anchor).
- Covers are positives, not anchors → **you do NOT need to generate N_sim for covers**.

An augmented original (e.g., pitch-shifted or tempo-stretched) sits structurally very close in embedding space to the unaugmented original — making it the ideal hard negative.

---

## Phase 1 — N_sim CQT Feature Generation

**New script: `indian_vocal_pipeline/prepare_data_nsim.py`**

1. Read the same train audio from `../data/train/indian_vocal_train`
2. For each **original** audio file (filename ending in `-o`):
   - Randomly pick **one** of the 4 augmentations (seeded with `random.seed(0)` for reproducibility):
     - pitch+4 semitones
     - pitch−4 semitones
     - tempo×1.2
     - tempo×0.8
   - Load audio with librosa, apply the chosen transformation, extract CQT using the same parameters as `prepare_data.py`
   - Save to `indian_vocal_pipeline/nsim_cqt_features/{id}_nsim.npy` (e.g. `100-0-ci-o_nsim.npy`)
3. Save `nsim_aug_map.json` mapping `{original_id} → chosen_aug_type` for reproducibility

---

## Phase 2 — Modified Triplet Loss Training

**Modify `indian_vocal_pipeline/main.py`:**

### 2a. Pre-load N_sim tensors before training loop
Build a dict `{clique_id: nsim_tensor}` by loading all `nsim_cqt_features/*.npy`, running them through the model once, and caching the embeddings. This is static — computed once before epoch 1.

### 2b. New `modified_triplet_loss` function
```python
def modified_triplet_loss(anchor, positive, neg_rand, neg_sim, margin=0.3):
    d_pos  = 1 - F.cosine_similarity(anchor, positive)
    d_rand = 1 - F.cosine_similarity(anchor, neg_rand)
    d_sim  = 1 - F.cosine_similarity(anchor, neg_sim)
    loss = torch.clamp(margin + 2 * d_pos - (d_sim + d_rand), min=0.0)
    return loss.mean()
```

### 2c. New `create_mod_triplets` function
Like `create_triplets` but returns 4-tuples `(anchor, positive, N_rand, N_sim)`. The N_sim embedding is looked up from the pre-loaded dict by clique label.

```python
def create_mod_triplets(embeddings, labels, is_orig, nsim_emb_dict, device):
    # For each original anchor:
    #   - positive  = random cover of the same clique
    #   - neg_rand  = random embedding from a different clique
    #   - neg_sim   = nsim_emb_dict[clique_label]  (pre-loaded augmented original)
```

### 2d. New `fine_tune_model_modified_triplet` training function
Same structure as existing `fine_tune_model` but:
- Uses `modified_triplet_loss` instead of `nn.TripletMarginLoss`
- Fetches N_sim from the pre-loaded dict per batch
- Saves checkpoints to `check_points/CQTNet_modified_triplet_best_...`

### 2e. Wire in via flag
`--modified_triplet=True` → calls `fine_tune_model_modified_triplet` instead of `fine_tune_model`

---

## Relevant Files

- `indian_vocal_pipeline/prepare_data.py` — template for CQT extraction (reuse parameters)
- `indian_vocal_pipeline/cqt_loader.py` — no changes needed
- `indian_vocal_pipeline/main.py` — add modified triplet functions + flag

---

## Key Decisions

| Decision | Recommendation |
|---|---|
| N_sim for originals only vs. all data points | Originals only (anchors = originals always) |
| One random aug per original vs. all 4 | One random per original (matches request; avoids over-complication) |
| Static (pre-computed once) vs. dynamic per epoch | Static — simpler; dynamic would require re-embedding all train data each epoch |
| Where to load N_sim | In `main.py` as a pre-loaded embedding dict, not in the DataLoader |

---

## Verification Steps

1. Run `prepare_data_nsim.py` → confirm `nsim_cqt_features/` has exactly one `.npy` per training original
2. Inspect `nsim_aug_map.json` → confirm each original maps to exactly one of the 4 aug types
3. Train for 5 epochs with `--modified_triplet=True` → confirm loss decreases and no "no valid triplets" warnings appear
4. Compare validation MAP against the standard triplet loss checkpoint to verify improvement

## Two Approaches for N_sim

---

### Approach A — Augmented Original (current plan)
N_sim = one randomly augmented version of the **same** original (pitch±4, tempo×1.2/0.8)

### Approach B — True Hard Negative Mining
N_sim = the non-cover training song with the **smallest cosine distance** to the anchor, found by:
```python
# embed all training songs once
# for each original anchor with clique C:
nsim = argmin_{songs where clique != C} cosine_distance(anchor_emb, song_emb)
```
Feasible because you only have ~126 training originals — a 126×126 distance matrix is trivial.

---

### Comparison

| | Approach A (Augmented) | Approach B (Hard Mining) |
|---|---|---|
| **What N_sim actually is** | The anchor song disguised slightly | A real, different, genuinely confusable song |
| **Alignment with paper's motivation** | Weak — paper says "songs from the same musical tradition that sound similar" | Strong — this IS what the paper means |
| **Computational cost** | Very cheap — offline, one-time | Cheap for 126 songs; needs re-computation if embeddings are refreshed |
| **Difficulty of the negative** | Unrealistically hard — it's literally the same song | Calibrated to the actual embedding space |
| **Risk of loss collapse** | High — d(anchor, N_sim) may be near 0, making the loss term dominate badly | Lower — distance reflects real confusion in the model |
| **Generalization** | Model learns "not my own augmentation" | Model learns "not this melodically similar Indian song" |
| **Reproducibility** | Trivial (seeded once) | Depends on when you re-embed; seed-controlled |
| **Staleness** | Never stale (CQT doesn't change) | Stale if not refreshed as model improves |

---

### Verdict: **Approach B is better for your use case**

The paper's explicit motivation is Indian songs that are melodically similar across different cliques (same raga, similar style, different song). Approach A teaches the model to separate a song from its own pitch-shifted copy — a completely different and arguably easier task. 

With only 126 training songs, you can:
1. Embed all training songs once before the fine-tuning loop starts
2. Build the hardest-negative lookup dict in one matrix operation
3. Optionally **refresh every 10–20 epochs** as the embeddings drift — still cheap

The augmented original approach is a useful fallback only if your training set were tiny (< 20 songs) and you had no real confusable negatives to mine, which is not your situation.