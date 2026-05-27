# Adapting Cover Song Identification to Indian Popular Music

This repository implements the research paper **"Adapting Cover Song Identification to Indian Popular Music"** (ISMIR 2025). It adapts CQTNet — a CNN originally trained on Western music (SHS100K) — to identify cover songs in Indian popular music using transfer learning, vocal separation, data augmentation, and a novel modified triplet loss.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Data Setup](#data-setup)
5. [Model Architecture (CQTNet)](#model-architecture-cqtnet)
6. [CQT Feature Extraction](#cqt-feature-extraction)
7. [Vocal Separation](#vocal-separation)
8. [Data Augmentation](#data-augmentation)
9. [Loss Functions Explained](#loss-functions-explained)
10. [N_sim Lookup (Hardest Negative Mining)](#n_sim-lookup-hardest-negative-mining)
11. [Pipelines](#pipelines)
    - [Indian Source Audio Pipeline](#1-indian-source-audio-pipeline-indian_new_pipeline)
    - [Indian Vocal Pipeline](#2-indian-vocal-pipeline-indian_vocal_pipeline)
    - [SHS100K Source Audio Pipeline](#3-shs100k-source-audio-pipeline-shs100k_new_pipeline)
    - [SHS100K Vocal Pipeline](#4-shs100k-vocal-pipeline-shs100k_vocal_pipeline)
    - [Legacy Pipeline (root main.py)](#5-legacy-pipeline-root-mainpy)
12. [Run Modes and Flags](#run-modes-and-flags)
13. [Evaluation Metrics](#evaluation-metrics)
14. [Results](#results)
15. [Checkpoints](#checkpoints)
16. [Troubleshooting](#troubleshooting)

---

## Overview

The goal is **cover song identification (CSI)**: given a query song, retrieve its cover versions from a database. The key insight of this paper is that **Indian popular music identity lies primarily in the vocal melody**, so vocal separation dramatically boosts performance on Indian datasets, while actually hurting performance on Western datasets.

**Training pipeline summary:**

```
Raw Audio (.mp3)
     │
     ├── (optional) Vocal Separation via Demucs ──► Vocal-only Audio
     │
     ▼
CQT Feature Extraction (librosa) ──► .npy arrays (84 freq bins × T time frames)
     │
     ▼
Stage 1: Cross-Entropy Transfer Learning (fine-tune CQTNet on new dataset)
     │
     ▼
Stage 2: Triplet Loss OR Modified Triplet Loss fine-tuning (embedding space optimization)
     │
     ▼
Evaluation: MAP, Top10, Rank1 on test set
```

---

## Project Structure

```
PP4/
├── config.py                  # Global training config (batch_size, lr, etc.)
├── main.py                    # Legacy training script (root-level)
├── utility.py                 # MAP calculation, normalization helpers
├── cqt_loader.py              # Legacy dataset loader
├── gencqt.py                  # Standalone CQT generator for raw audio
├── requirements.txt           # Python dependencies
├── run_all.sh                 # Legacy run script
│
├── models/
│   ├── __init__.py            # Exports CQTNet
│   ├── basic_module.py        # Base class with save/load methods
│   └── CQTNet.py              # The CNN model architecture
│
├── data/                      # Datasets (NOT in git — see Data Setup below)
│   ├── indian_train.csv       # Metadata: clique, version, title, type, id
│   ├── indian_test.csv
│   ├── shs100k-train.csv      # Metadata: clique, version, filename, label
│   ├── shs100k-test.csv
│   ├── train/
│   │   ├── indian/            # Indian source audio (.mp3 files)
│   │   ├── indian_vocal_train/# Indian vocal-separated audio (.mp3)
│   │   ├── shs100k_train/     # SHS100K source audio (.mp3)
│   │   └── shs100k_train_vocal/ # SHS100K vocal-separated audio (.mp3)
│   └── test/
│       ├── indian/            # Indian test source audio
│       ├── indian_vocal/      # Indian test vocal-separated audio
│       ├── shs100k-test/      # SHS100K test source audio
│       └── shs100k_test_vocal/# SHS100K test vocal-separated audio
│
├── check_points/              # Saved model weights (.pth files)
│
├── scripts/
│   └── vocalseparation/
│       ├── separate_vocals.py # Demucs-based vocal separation
│       └── run_vocal_sep.sh   # Batch vocal separation for all datasets
│
├── indian_new_pipeline/       # Indian source audio pipeline
│   ├── main.py                # Training script
│   ├── cqt_loader.py          # IndianCQT dataset class
│   ├── prepare_data.py        # Extract CQT + generate split files
│   ├── compute_nsim_lookup.py # Precompute hardest negatives
│   ├── run_all_indian.sh      # End-to-end run script
│   ├── nsim_lookup.json       # Precomputed N_sim mappings
│   ├── indian_train.txt       # Train split: filename,clique
│   ├── indian_test.txt        # Test split: filename,clique
│   └── cqt_features/          # Generated .npy CQT arrays (gitignored)
│
├── indian_vocal_pipeline/     # Indian vocal-separated pipeline
│   ├── main.py
│   ├── cqt_loader.py          # IndianVocalCQT dataset class
│   ├── prepare_data.py
│   ├── compute_nsim_lookup.py
│   ├── run_all_indian_vocal.sh
│   ├── nsim_lookup.json
│   ├── indian_train.txt
│   ├── indian_test.txt
│   └── cqt_features/          # (gitignored)
│
├── shs100k_new_pipeline/      # SHS100K source audio pipeline
│   ├── main.py
│   ├── cqt_loader.py          # SHS100kCQT dataset class
│   ├── prepare_data.py
│   ├── generate_cqt.py
│   ├── compute_nsim_lookup.py
│   ├── run_all_shs.sh
│   ├── nsim_lookup.json
│   ├── shs100k_train.txt      # Train split: filename,clique,is_original
│   ├── shs100k-test.txt       # Test split: filename,clique,is_original
│   └── cqt_features/          # (gitignored)
│
└── shs100k_vocal_pipeline/    # SHS100K vocal-separated pipeline
    ├── main.py
    ├── cqt_loader.py          # SHS100kVocalCQT dataset class
    ├── prepare_data.py
    ├── generate_cqt.py
    ├── compute_nsim_lookup.py
    ├── run_all_shs_vocal.sh
    ├── nsim_lookup.json
    ├── shs100k_train.txt
    ├── shs100k-test.txt
    └── cqt_features/          # (gitignored)
```

---

## Prerequisites

### Hardware

- **GPU**: NVIDIA GPU with CUDA support (training is very slow on CPU)
- **RAM**: 16 GB minimum, 32 GB recommended for SHS100K
- **Disk**: ~50 GB for SHS100K audio + CQT features; ~5 GB for Indian dataset

### Software

- Python 3.8+
- CUDA-compatible PyTorch

### Install Dependencies

```bash
cd PP4
pip install -r requirements.txt
```

The `requirements.txt` contains:
```
torch
numpy
librosa
tqdm
torchvision
pydub
torchnet
fire
```

**Additional packages needed depending on what you run:**

```bash
# For vocal separation (Demucs)
pip install demucs

# For data preparation scripts
pip install pandas

# For SHS100K data prep (already in requirements)
pip install librosa
```

---

## Data Setup

**The following files are gitignored and must be provided manually.** This is the most important step — the code will not work without them.

### What You Need to Provide

| Item | Location | Description |
|------|----------|-------------|
| Indian audio (train) | `data/train/indian/*.mp3` | Source audio for Indian training songs |
| Indian audio (test) | `data/test/indian/*.mp3` | Source audio for Indian test songs |
| Indian metadata | `data/indian_train.csv`, `data/indian_test.csv` | CSV with columns: `clique, version, title, type, id` |
| SHS100K audio (train) | `data/train/shs100k_train/*.mp3` | SHS100K training audio files |
| SHS100K audio (test) | `data/test/shs100k-test/*.mp3` | SHS100K test audio files |
| SHS100K metadata | `data/shs100k-train.csv`, `data/shs100k-test.csv` | CSV with columns: `clique, version, filename, label` |
| Pre-trained checkpoints | `check_points/*.pth` | Pre-trained model weights (see [Checkpoints](#checkpoints)) |

### Indian CSV Format

```csv
clique,version,title,type,id
100,1,Do Lafzon Ki Hai Dil Ki Kahani,cover,100-1-ci-c
100,0,Do Lafzon Ki Hai Dil Ki Kahani,original,100-0-ci-o
```

- `clique`: A numeric ID grouping an original song and all its covers together
- `version`: 0 = original, 1+ = cover
- `id`: Filename stem (the `.mp3` file is named `{id}.mp3`)

### SHS100K CSV Format

```csv
clique,version,filename,label
1004,0,1004_0.mp3,19
1004,1,1004_1.mp3,19
```

- `clique`: Group ID for original + covers
- `version`: Lowest version number within a clique = original
- `filename`: Actual audio filename

### Indian Song Naming Convention

Filenames follow the pattern: `{clique}-{version}-ci-{type}`

- `100-0-ci-o` → clique 100, version 0, original (`-o` suffix)
- `100-1-ci-c` → clique 100, version 1, cover (`-c` suffix)

The code uses the `-o` suffix (or `-o-` in the middle) to detect originals. This is important for triplet loss where anchors must be original tracks.

### Generated Files (gitignored — will be recreated by scripts)

| Item | Generated By |
|------|-------------|
| `*/cqt_features/*.npy` | `prepare_data.py` or `generate_cqt.py` in each pipeline |
| `*/indian_train.txt`, `*/indian_test.txt` | `prepare_data.py` |
| `*/shs100k_train.txt`, `*/shs100k-test.txt` | `prepare_data.py` |
| Vocal-separated audio | `scripts/vocalseparation/separate_vocals.py` |

---

## Model Architecture (CQTNet)

CQTNet is a CNN designed for cover song identification. It takes a CQT spectrogram as input and produces a 300-dimensional embedding.

```
Input: (1, 84, 400) — 1 channel, 84 frequency bins, 400 time frames

Layer Stack:
  conv0 (1→32, 12×3, dilation=1×1) → BN → ReLU
  conv1 (32→64, 13×3, dilation=1×2) → BN → ReLU → MaxPool(1×2)
  conv2 (64→64, 13×3) → BN → ReLU
  conv3 (64→64, 3×3, dilation=1×2) → BN → ReLU → MaxPool(1×2)
  conv4 (64→128, 3×3) → BN → ReLU
  conv5 (128→128, 3×3, dilation=1×2) → BN → ReLU → MaxPool(1×2)
  conv6 (128→256, 3×3) → BN → ReLU
  conv7 (256→256, 3×3, dilation=1×2) → BN → ReLU → MaxPool(1×2)
  conv8 (256→512, 3×3) → BN → ReLU
  conv9 (512→512, 3×3, dilation=1×2) → BN → ReLU

  AdaptiveMaxPool2d → (512, 1, 1)
  FC0: 512 → 300 (embedding layer)
  FC1: 300 → num_classes (classification head)

Output: (classification_scores, 300-dim embedding)
```

**Key design choices:**
- Dilated convolutions capture pitch patterns across frequency ranges
- The first conv layer (`conv0`) is frozen during fine-tuning (it captures low-level audio patterns that transfer well)
- `FC0` output is the **embedding** used for similarity matching at inference
- `FC1` is the **classification head** — its size is changed to match the number of unique songs in each dataset

---

## CQT Feature Extraction

CQT (Constant-Q Transform) converts audio into a time-frequency representation where frequency bins are logarithmically spaced — matching how musical notes are organized.

**How it works in this project:**

```python
data, sr = librosa.load(audio_path)       # Load audio at 22050 Hz
cqt = np.abs(librosa.cqt(y=data, sr=sr))  # Compute CQT magnitude
# Downsample time axis by averaging every 20 frames
mean_size = 20
new_cqt = np.zeros((height, length // mean_size))
for i in range(length // mean_size):
    new_cqt[:, i] = cqt[:, i*20:(i+1)*20].mean(axis=1)
np.save(output_path, new_cqt)             # Save as .npy
```

**At training time**, each `.npy` is loaded and preprocessed:
1. Normalize: divide by `max(abs(values))`
2. Pad or truncate to exactly **(84 frequency bins, 400 time frames)**
3. Add a channel dimension → shape `(1, 84, 400)`

---

## Vocal Separation

**Why:** Indian popular music identity is primarily carried by the vocal melody. Removing instrumental accompaniment helps the model focus on what makes a cover song identifiable.

**How:** Uses Meta's **Demucs** (htdemucs model) to separate vocals from the audio mix.

```bash
# Install demucs
pip install demucs

# Separate vocals for all datasets at once
bash scripts/vocalseparation/run_vocal_sep.sh

# Or run individually
python scripts/vocalseparation/separate_vocals.py --dataset indian_test
python scripts/vocalseparation/separate_vocals.py --dataset shs100k_train
python scripts/vocalseparation/separate_vocals.py --dataset shs100k_test
```

**Input/Output mapping:**

| Input Directory | Output Directory |
|----------------|-----------------|
| `data/test/indian/` | `data/test/indian_vocal/` |
| `data/train/shs100k_train/` | `data/train/shs100k_train_vocal/` |
| `data/test/shs100k-test/` | `data/test/shs100k_test_vocal/` |

**Note:** Indian training vocal separation goes to `data/train/indian_vocal_train/`. You may need to run vocal separation on the training Indian audio separately or adjust paths.

**Key finding:** Vocal separation improves Indian CSI significantly (MAP: 0.7238 → 0.8629) but **hurts** Western CSI (MAP: 0.8768 → 0.7181). This confirms the cultural difference in how cover songs preserve identity.

---

## Data Augmentation

The CoversIndian dataset has only **one cover per original**. To increase training diversity, **4 augmented variants** are created from each cover:

| Variant | Transformation | Parameter |
|---------|---------------|-----------|
| 1 | Pitch shift up | +4 semitones |
| 2 | Pitch shift down | -4 semitones |
| 3 | Tempo stretch fast | ×1.2 speed |
| 4 | Tempo stretch slow | ×0.8 speed |

This gives **6 samples per song group**: 1 original + 1 real cover + 4 augmented covers.

The augmented samples appear in the training data as additional entries with the same clique ID. The batch structure groups them together (batch_size=60 → 10 groups of 6).

---

## Loss Functions Explained

The training process uses **two stages** with different loss functions.

### Stage 1: Cross-Entropy Loss (Classification)

The model learns to classify each input CQT into the correct "clique" (song group).

```
Loss = CrossEntropy(model_output, clique_label)
```

- `model_output`: Scores from the final FC layer (one score per clique)
- `clique_label`: Integer ID of the song's clique

This teaches the model to recognize songs by their class label. The penultimate layer (FC0) naturally learns to produce similar embeddings for songs in the same clique.

### Stage 2 Option A: Standard Triplet Loss

After cross-entropy training, the model is further fine-tuned with triplet loss to directly optimize the embedding space geometry.

**Components:**
- **Anchor (A)**: Embedding of an **original** song
- **Positive (P)**: Embedding of a **cover** of that song (same clique)
- **Negative (N_rand)**: Embedding of a **random song from a different clique**

**Formula:**

```
L = max(0, margin + d(A, P) - d(A, N_rand))
```

where `d(x, y) = 1 - cosine_similarity(x, y)` and `margin = 0.05`.

**What it does:** Pushes covers closer to their originals while pushing non-covers away. The margin ensures a minimum gap between positive and negative distances.

**How triplets are created in code:**
1. For each original track in the batch (detected by `-o` in filename):
   - Pick a random cover from the same clique as the positive
   - Pick a random song from a different clique as the negative
2. If no valid triplets exist in the batch (e.g., no originals), skip that batch

### Stage 2 Option B: Modified Triplet Loss (Paper's Novel Contribution)

Standard triplet loss uses a **random** negative, which is often too easy — the model already knows it's different. In Indian music, many songs from different cliques sound melodically similar. The modified loss adds a **hard negative** to force the model to learn finer distinctions.

**Components:**
- **Anchor (A)**: Original song embedding
- **Positive (P)**: Cover song embedding (same clique)
- **N_rand**: Random negative (different clique)
- **N_sim**: **Hardest negative** — the non-cover song **most similar** to the anchor in the dataset

**Formula:**

```
L = max(0, margin + 2·d(A, P) - [d(A, N_sim) + d(A, N_rand)])
```

where `margin = 0.3`.

**Why 2× on the positive distance?** Because there are two negative terms. The factor of 2 balances the equation so the positive distance is weighed equally against the average of the two negative distances.

**How N_sim is found:** See the next section.

---

## N_sim Lookup (Hardest Negative Mining)

The N_sim lookup maps each song to its **most similar song from a different clique** — the hardest non-cover negative.

### How It's Computed (`compute_nsim_lookup.py`)

```
1. Load all CQT features (.npy files) from the dataset
2. Normalize each to shape (84, 400), then flatten to a vector
3. L2-normalize all vectors
4. Compute the cosine similarity matrix (N × N)
5. For each song:
   a. Mask out all songs from the SAME clique (set similarity to -∞)
   b. Find the song with the HIGHEST cosine similarity = N_sim
6. Save the mapping to nsim_lookup.json
```

### nsim_lookup.json Format

```json
{
  "100-0-ci-o": {
    "nsim_id": "205-1-ci-c",
    "nsim_clique": "205",
    "cosine_similarity": 0.8934
  }
}
```

Each entry says: "For song `100-0-ci-o`, the most similar song from a different clique is `205-1-ci-c` with cosine similarity 0.8934."

### How N_sim Is Used During Training

1. **Before training starts**, the model computes embeddings for all N_sim songs and stores them in a dictionary keyed by clique label.
2. **During each batch**, when creating modified triplets:
   - The anchor's clique label is looked up in the N_sim embedding dict
   - If found, that embedding becomes `neg_sim`
   - If not found, `neg_rand` is used as fallback
3. **Every 10 epochs**, N_sim embeddings are **refreshed** by re-running the model on the N_sim songs. This is important because the model's embeddings change during training.

### When to Run compute_nsim_lookup.py

You **must** run this script **before** using `--modified_triplet=True`. Run it from inside each pipeline directory:

```bash
cd indian_new_pipeline
python compute_nsim_lookup.py

cd ../indian_vocal_pipeline
python compute_nsim_lookup.py

cd ../shs100k_new_pipeline
python compute_nsim_lookup.py

cd ../shs100k_vocal_pipeline
python compute_nsim_lookup.py
```

**Important:** The nsim_lookup.json is computed on **raw CQT features** (not model embeddings). It only needs to be recomputed if your CQT features change.

---

## Pipelines

Each pipeline is a self-contained directory with its own `main.py`, `cqt_loader.py`, `prepare_data.py`, and run script. They all share the same `models/`, `config.py`, and `utility.py` from the root.

### 1. Indian Source Audio Pipeline (`indian_new_pipeline/`)

**Dataset:** CoversIndian with source (non-separated) audio.

**Step-by-step:**

```bash
# Step 1: Prepare CQT features and split files
cd PP4/indian_new_pipeline
python prepare_data.py

# Step 2: (Optional) Compute N_sim lookup for modified triplet loss
python compute_nsim_lookup.py

# Step 3: Train — choose ONE of these modes:

# Mode A: Cross-Entropy only
python main.py transfer_learning

# Mode B: Cross-Entropy → Triplet Loss
python main.py transfer_learning --fine_tune=True

# Mode C: Cross-Entropy → Modified Triplet Loss
python main.py transfer_learning --modified_triplet=True

# Mode D: Skip CE, go straight to Triplet (from existing checkpoint)
python main.py transfer_learning --fine_tune=True --skip_transfer=True

# Mode E: Skip CE, go straight to Modified Triplet
python main.py transfer_learning --modified_triplet=True --skip_transfer=True
```

**Or run the full pipeline at once:**

```bash
bash indian_new_pipeline/run_all_indian.sh
```

This runs: `prepare_data.py` → `main.py transfer_learning --fine_tune=True`

### 2. Indian Vocal Pipeline (`indian_vocal_pipeline/`)

**Dataset:** CoversIndian with vocal-separated audio. This is the **best-performing pipeline** per the paper results.

**Step-by-step:**

```bash
# Step 0: Separate vocals (if not already done)
bash scripts/vocalseparation/run_vocal_sep.sh
# (or just the Indian datasets)

# Step 1: Prepare CQT features from vocal audio
cd PP4/indian_vocal_pipeline
python prepare_data.py

# Step 2: (Optional) Compute N_sim lookup
python compute_nsim_lookup.py

# Step 3: Train (same modes as Indian pipeline above)
python main.py transfer_learning --fine_tune=True
python main.py transfer_learning --modified_triplet=True
# etc.
```

**Or run the full pipeline:**

```bash
bash indian_vocal_pipeline/run_all_indian_vocal.sh
```

### 3. SHS100K Source Audio Pipeline (`shs100k_new_pipeline/`)

**Dataset:** SHS100K (large Western cover song dataset) with source audio.

**Step-by-step:**

```bash
# Step 1: Prepare text split files
cd PP4/shs100k_new_pipeline
python prepare_data.py

# Step 2: Generate CQT features
python generate_cqt.py

# Step 3: (Optional) Compute N_sim lookup
python compute_nsim_lookup.py

# Step 4: Train
python main.py transfer_learning
python main.py transfer_learning --fine_tune=True
python main.py transfer_learning --modified_triplet=True
python main.py transfer_learning --fine_tune=True --skip_transfer=True
# etc.
```

**Or run the full pipeline:**

```bash
bash shs100k_new_pipeline/run_all_shs.sh
```

**Note:** SHS100K has a separate `generate_cqt.py` because `prepare_data.py` only generates the text split files, not the CQT features.

### 4. SHS100K Vocal Pipeline (`shs100k_vocal_pipeline/`)

**Dataset:** SHS100K with vocal-separated audio.

```bash
# Step 0: Separate vocals
bash scripts/vocalseparation/run_vocal_sep.sh

# Step 1-4: Same as SHS100K source pipeline
cd PP4/shs100k_vocal_pipeline
python prepare_data.py
python generate_cqt.py
python compute_nsim_lookup.py  # if using modified triplet
python main.py transfer_learning --fine_tune=True
```

**Or:**

```bash
bash shs100k_vocal_pipeline/run_all_shs_vocal.sh
```

### 5. Legacy Pipeline (root `main.py`)

The root `main.py` is the original training script before the project was refactored into separate pipelines. It works with the `cqt_loader.py` in the root directory and reads from `indian_new_pipeline/cqt_features/`.

```bash
# Legacy mode (from project root)
python main.py transfer_learning --fine_tune=True --skip_transfer=True
```

**Use the pipeline-specific scripts instead** — they are cleaner, better organized, and produce structured run logs.

---

## Run Modes and Flags

All pipeline `main.py` scripts accept these flags via Python Fire:

| Flag | Default | Description |
|------|---------|-------------|
| `--fine_tune=True` | `False` | After Stage 1 (CE), run Stage 2 with **Triplet Loss** |
| `--modified_triplet=True` | `False` | After Stage 1 (CE), run Stage 2 with **Modified Triplet Loss** |
| `--skip_transfer=True` | `False` | Skip Stage 1 (CE) entirely. Use the loaded checkpoint directly and go to Stage 2 |
| `--load_model_path="..."` | varies | Path to a specific checkpoint to start from |
| `--run_name="..."` | auto-generated | Custom name for this training run (used in log/checkpoint filenames) |

### Common Combinations

| What You Want | Command |
|---------------|---------|
| Only Cross-Entropy fine-tuning | `python main.py transfer_learning` |
| CE → Triplet Loss | `python main.py transfer_learning --fine_tune=True` |
| CE → Modified Triplet Loss | `python main.py transfer_learning --modified_triplet=True` |
| Skip CE, Triplet only (resume) | `python main.py transfer_learning --fine_tune=True --skip_transfer=True` |
| Skip CE, Modified Triplet only | `python main.py transfer_learning --modified_triplet=True --skip_transfer=True` |
| Custom checkpoint | `python main.py transfer_learning --load_model_path="check_points/my_model.pth"` |

### Training Hyperparameters

| Parameter | Stage 1 (CE) | Stage 2 (Triplet) | Stage 2 (Mod Triplet) |
|-----------|-------------|-------------------|----------------------|
| Epochs | 100 | 75 | 75 |
| Learning Rate | 1e-4 | 1e-4 | 1e-4 |
| Batch Size | 120 (Indian), 240 (SHS100K) | Same | Same |
| Margin | N/A | 0.05 | 0.3 |
| Early Stop Patience | 15 | 15 | 15 |
| Frozen Layers | conv0 | conv0 | conv0 |
| N_sim Refresh | N/A | N/A | Every 10 epochs |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAP** (Mean Average Precision) | Primary metric. Measures average precision across all queries. Higher = better. |
| **Top10** | Precision@10 — fraction of correct covers found in the top 10 results. Range [0, 1]. |
| **Rank1** | Mean rank of the first correctly identified cover. Lower = better. |

**How evaluation works:**
1. Pass all test songs through the model to get 300-dim embeddings
2. L2-normalize all embeddings
3. Compute pairwise cosine similarity (as negative distance matrix)
4. For each song, rank all other songs by similarity
5. Calculate MAP, Top10, Rank1 from the rankings

---

## Results

### CoversIndian Test Set (Fine-tuned CQTNet)

| Method | MAP | Top10 | Rank1 |
|--------|-----|-------|-------|
| Baseline CQTNet | 0.7238 | 0.0840 | 8 |
| Fine-tuned CQTNet | 0.7379 | 0.0819 | 11 |
| VocalSep | 0.9001 | 0.0962 | 2 |
| DA + VocalSep | 0.9131 | 0.0958 | 2 |
| Triplet + DA + VocalSep | 0.8953 | 0.0971 | 3 |
| **Mod Triplet + DA + VocalSep** | **0.9234** | **0.0975** | **2** |

### Covers80 (Western Dataset)

| Method | MAP | Top10 | Rank1 |
|--------|-----|-------|-------|
| Baseline CQTNet | 0.8768 | 0.0944 | 3 |
| Baseline CQTNet + VocalSep | 0.7181 | 0.0819 | 8 |

Vocal separation **hurts** Western cover song identification — confirming that the approach is specifically tuned for Indian music.

---

## Checkpoints

### Required Pre-trained Checkpoints

These must be present in `check_points/` before training:

| File | Description | Used By |
|------|-------------|---------|
| `cqtnet_indian_best.pth` | CQTNet pre-trained on SHS100K, head adapted for Indian classes | Indian pipelines (default init) |
| `cqtnet_shs100k_best.pth` | CQTNet pre-trained on SHS100K (original weights) | SHS100K pipelines (default init) |

### Checkpoints Generated During Training

Training produces checkpoints with names like:
- `CQTNet_indian_ce_best_{run_name}.pth` — Best model from CE stage
- `CQTNet_indian_triplet_best_{run_name}.pth` — Best model from Triplet stage
- `CQTNet_indian_mod_triplet_best_{run_name}.pth` — Best from Modified Triplet stage

Similarly for vocal and SHS100K pipelines.

### Training Logs

Pipeline runs produce JSON logs in `{pipeline}/logs/` containing:
- Per-epoch loss and validation metrics
- Best epoch and checkpoint path
- Final test set performance
- All hyperparameters

---

## Troubleshooting

### "FileNotFoundError: nsim_lookup.json not found"

You tried to use `--modified_triplet=True` without first computing the N_sim lookup. Run:

```bash
cd <pipeline_directory>
python compute_nsim_lookup.py
```

### "No valid triplets in this batch"

This means the batch didn't contain both originals and covers for any clique. This can happen with small batch sizes or unbalanced data. The training loop skips these batches. If it happens too often, increase batch size.

### CQT feature shape mismatches

All CQT features are padded/truncated to `(84, 400)` at load time. If you see shape errors, ensure your audio files are valid and long enough (> 1000 samples at 22050 Hz).

### Out of GPU memory

Reduce batch size:

```bash
python main.py transfer_learning --batch_size=60
```

### "Checkpoint not found" errors

Ensure the required pre-trained checkpoints exist in `check_points/`. See the [Checkpoints](#checkpoints) section.

### Running on a different machine

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt && pip install pandas demucs`
3. Place audio files in the correct `data/` subdirectories
4. Place CSV metadata files in `data/`
5. Place pre-trained checkpoints in `check_points/`
6. (Optional) Run vocal separation: `bash scripts/vocalseparation/run_vocal_sep.sh`
7. Run the desired pipeline's data prep: `python prepare_data.py` (and `generate_cqt.py` for SHS100K)
8. (Optional) Compute N_sim: `python compute_nsim_lookup.py`
9. Run training: `python main.py transfer_learning --fine_tune=True`

### File paths are hardcoded

Some shell scripts have absolute paths like `/teamspace/studios/this_studio/PP4/`. Update these to match your environment, or simply `cd` into the pipeline directory and run the Python scripts directly.
