# Adapting Cover Song Identification to Indian Popular Music
### ISMIR 2025 Submission

---

## 1. Methodology

### 1.1 CQTNet (Baseline Model)

**Architecture:**
- Input: Constant-Q Transform (CQT) representation of audio, shape `(frequency_bins, time_frames)`
- Backbone: CNN with **dilated convolutions** in early layers to capture pitch patterns across frequency ranges
- Specialised filter designs in the first layers to handle musical structure
- Followed by fully connected (FC) layers
- Final layer: classification head (number of nodes = number of unique original songs)
- Loss during training: **Cross-Entropy Loss** (label classification task)

**Training objective:**
```
loss = CrossEntropyLoss(FC_output, song_class_label)
```

**Inference (embedding retrieval):**
```
embedding = model.penultimate_layer(cqt_input)   # extract before final FC
similarity = 1 - cosine_distance(embedding_A, embedding_B)
ranking = argsort(cosine_distance(query_embedding, all_embeddings))
```

**Pre-trained checkpoint:** Trained on **SHS100K** (large Western cover song dataset).  
Benchmarked MAP: 0.917 (YouTube350), 0.933 (Mazurkas).

---

### 1.2 Fine-Tuning with CoversIndian Dataset

**Motivation:** Cultural invariants exist between original and cover songs. Fine-tuning transfers
knowledge from Western pre-training while adapting to Indian cover song characteristics.

**Steps:**
1. Load pre-trained CQTNet weights (trained on SHS100K).
2. Replace the final classification layer:
   - Original node count = number of SHS100K classes
   - New node count = number of CoversIndian classes (smaller)
3. Fine-tune all layers on CoversIndian train split using Cross-Entropy Loss.
4. Keep the same input format: CQT features of audio.

```python
model.classifier = nn.Linear(in_features, num_indian_classes)
optimizer = Adam(model.parameters())
for epoch in range(200):
    for cqt, label in train_loader:
        logits = model(cqt)
        loss = cross_entropy(logits, label)
        loss.backward()
        optimizer.step()
```

> All subsequent experiments use this fine-tuned CQTNet as the base model.

---

### 1.3 Vocal Separation

**Motivation:** Indian popular music identity lies primarily in the vocal melody. Removing
accompaniment instruments should boost similarity matching on vocal-centric covers.

**Steps:**
1. Apply vocal source separation to all audio files using the **music.ai (Moises) API**.
2. Retain only the separated vocal track.
3. Compute CQT on the vocal-only audio.
4. Use this vocal CQT as the model input instead of the full mix CQT.

```python
vocal_audio = music_ai_api.separate(audio, stem="vocals")
cqt_vocal = librosa.cqt(vocal_audio, sr=sr, ...)
```

**Applied to:** Both CoversIndian and Covers80 datasets (for comparison).

---

### 1.4 Data Augmentation

**Motivation:** CoversIndian has only one cover per original song. Augmentation generates
additional synthetic cover variants to broaden training diversity.

**Variants created per cover song (4 augmented + 1 original = 5 total):**

| Variant | Transformation       | Parameter     |
|---------|----------------------|---------------|
| 1       | Pitch shift up       | +4 semitones  |
| 2       | Pitch shift down     | −4 semitones  |
| 3       | Tempo stretch fast   | ×1.2          |
| 4       | Tempo stretch slow   | ×0.8          |

```python
import librosa

def augment(audio, sr):
    variants = [audio]  # original
    variants.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=4))
    variants.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-4))
    variants.append(librosa.effects.time_stretch(audio, rate=1.2))
    variants.append(librosa.effects.time_stretch(audio, rate=0.8))
    return variants
```

**Applied to:** Cover version audio (and vocal-separated cover audio).  
**Batch structure:** Batch size = 60 → processes 10 groups of 6 (1 original + 5 augmented covers).  
Shuffle = **False** to keep these groups intact.

---

### 1.5 Triplet Loss

**Motivation:** Cross-entropy trains similarity to a class label, but does not explicitly push
non-cover songs away. Triplet loss directly optimises the embedding space geometry.

**Triplet components:**
- `A_ori` — anchor: embedding of the **original song**
- `P_cover(i)` — positive: embedding of the **i-th cover** of that song
- `N_rand` — negative: embedding of a **randomly sampled non-cover** song

**Loss function:**
$$
L = \max\left(0,\ \alpha + d\left(A_{ori},\ P_{cover(i)}\right) - d\left(A_{ori},\ N_{rand}\right)\right)
$$

where $d(\cdot)$ is cosine distance and $\alpha$ is the margin hyperparameter.

```python
def triplet_loss(anchor, positive, negative, margin=0.3):
    d_pos = cosine_distance(anchor, positive)
    d_neg = cosine_distance(anchor, negative)
    loss = torch.clamp(margin + d_pos - d_neg, min=0.0)
    return loss.mean()
```

**Training:** Applied at the **penultimate embedding layer** (replaces cross-entropy at final layer).  
Fine-tuned for **150 epochs**.

---

### 1.6 Modified Triplet Loss

**Motivation:** In Indian music, negative cover instances (songs from the same musical tradition)
can be very melodically similar to the anchor. Using only a random negative is insufficient.
The modified loss includes the **most similar negative** to force the model to learn finer distinctions.

**Triplet components (extended):**
- `A_ori` — anchor: original song embedding
- `P_cover(i)` — positive: i-th cover embedding
- `N_rand` — random negative (non-cover)
- `N_sim` — hardest negative: the non-cover song **closest in embedding space** to the anchor

**Loss function:**
$$
L = \max\left(0,\ \alpha + 2 \cdot d\left(A_{ori},\ P_{cover(i)}\right) - \left[d\left(A_{ori},\ N_{sim}\right) + d\left(A_{ori},\ N_{rand}\right)\right]\right)
$$

The factor of **2** on the positive distance balances the two negative terms.

```python
def modified_triplet_loss(anchor, positive, neg_rand, neg_sim, margin=0.3):
    d_pos  = cosine_distance(anchor, positive)
    d_rand = cosine_distance(anchor, neg_rand)
    d_sim  = cosine_distance(anchor, neg_sim)
    loss = torch.clamp(margin + 2 * d_pos - (d_sim + d_rand), min=0.0)
    return loss.mean()
```

**Two-stage fine-tuning procedure:**

```
Stage 1 — Label Classification Fine-tune:
  M = fine_tune(CQTNet_pretrained, CoversIndian, loss=CrossEntropy, epochs=200)

Stage 2 — Modified Triplet Loss Fine-tune:
  E = get_embeddings(M, train_data)        # embed all (original, cover) pairs
  For each (A_orig, Cover_A) in E:
      C_A    = all covers of A_orig
      N_rand = random sample from E_cover where N_rand ∉ C_A
      N_sim  = argmin_{E_cover(i) ∉ C_A} dist(A_orig, E_cover(i))
      L      = modified_triplet_loss(A_orig, Cover_A, N_rand, N_sim)
  Update model parameters on L  (epochs=150)
```

---

### 1.7 CQTNet + MERT / HuBERT Embedding Fusion

**Motivation:** Explore whether large self-supervised audio models can complement CQT features
for cover song identification.

**Models used:**
- **HuBERT-large** — self-supervised speech model, adapted for general audio
- **MERT-330M** — music-specific large audio model trained with self-supervised objectives
  (beat tracking, singer ID, music tagging, key detection, genre classification)

**Fusion architecture:**
```
Audio
  ├─► CQT features ──► CQTNet backbone ──► CQTNet embedding (D1)
  │
  └─► Raw waveform ──► MERT/HuBERT (frozen) ──► Feed-forward layers ──► Audio embedding (D2)

[CQTNet embedding (D1) || Audio embedding (D2)] ──► Final FC layer ──► Similarity score
```

```python
# MERT/HuBERT weights are FROZEN (non-trainable)
mert_out   = mert_model(waveform).last_hidden_state.mean(dim=1)   # pooled
mert_proj  = ff_layers(mert_out)          # trainable projection

cqt_emb    = cqtnet(cqt_input)            # from penultimate layer
fused      = torch.cat([cqt_emb, mert_proj], dim=-1)
output     = final_fc(fused)
```

**Similarity at inference:** Cosine similarity on the fused output embedding.

---

## 2. Experiments

### 2.1 Dataset

| Dataset        | Type         | Language          | Song Pairs | Usage                     |
|----------------|--------------|-------------------|------------|---------------------------|
| CoversIndian   | Indian       | Hindi + Malayalam | 245        | 126 train / 119 test      |
| Covers80       | Western      | English           | 80         | Test only (comparison)    |
| SHS100K        | Western      | —                 | Large       | Pre-training (CQTNet)     |

### 2.2 Experimental Setup

| Setting               | Value                                     |
|-----------------------|-------------------------------------------|
| CQT library           | Librosa                                   |
| Vocal separation      | music.ai (Moises) API                     |
| Augmentation library  | Librosa (pitch shift, time stretch)       |
| Hardware              | Intel Gaudi3 AI (128 GB), NVIDIA A100 (80 GB) |
| Random seed           | 0                                         |
| Label classification epochs | 200                               |
| Triplet loss epochs   | 150                                       |
| Batch size            | 60 (10 groups × 6: 1 original + 5 augmented covers) |
| Shuffle               | False                                     |

### 2.3 Evaluation Metrics

| Metric  | Description                                              |
|---------|----------------------------------------------------------|
| MAP     | Mean Average Precision — primary ranking metric          |
| Top10   | Precision@10 — fraction of correct covers in top 10     |
| Rank1   | Mean rank of the first correctly identified cover        |

---

## 3. Results

### 3.1 Baseline CQTNet vs Fine-tuned CQTNet (CoversIndian)

| Model                | MAP    | Top10  | Rank1 |
|----------------------|--------|--------|-------|
| Baseline CQTNet      | 0.7238 | 0.0840 | 8     |
| Fine-tuned CQTNet    | 0.7379 | 0.0819 | 11    |

---

### 3.2 Vocal Separation

| Dataset       | Method                        | MAP    | Top10  | Rank1 |
|---------------|-------------------------------|--------|--------|-------|
| CoversIndian  | Baseline CQTNet               | 0.7238 | 0.0840 | 8     |
| CoversIndian  | Baseline CQTNet + VocalSep    | 0.8629 | 0.0937 | 3     |
| CoversIndian  | Fine-tuned CQTNet             | 0.7379 | 0.0819 | 11    |
| CoversIndian  | Fine-tuned CQTNet + VocalSep  | 0.9001 | 0.0962 | 2     |
| Covers80      | Baseline CQTNet               | 0.8768 | 0.0944 | 3     |
| Covers80      | Baseline CQTNet + VocalSep    | 0.7181 | 0.0819 | 8     |

> Vocal separation consistently improves Indian cover song results but **degrades** Western (Covers80) results — confirming that vocal melody is a stronger identity signal in Indian popular music.

---

### 3.3 Data Augmentation (CoversIndian, Fine-tuned CQTNet)

| Method                    | MAP    | Top10  | Rank1 |
|---------------------------|--------|--------|-------|
| DA + Source Audio         | 0.7307 | 0.0857 | 9     |
| DA + VocalSep Audio       | 0.9131 | 0.0958 | 2     |

> Data augmentation improves results only when combined with vocal separation. No improvement on source audio.

---

### 3.4 Triplet Loss (CoversIndian, Fine-tuned CQTNet)

| Method                       | MAP    | Top10  | Rank1 |
|------------------------------|--------|--------|-------|
| DA + Source Audio            | 0.7696 | 0.0861 | 6     |
| DA + VocalSep Audio          | 0.9079 | 0.0975 | 2     |

> Triplet loss improves source audio results with augmentation but slightly degrades vocal-separated augmented data.

---

### 3.5 Modified Triplet Loss (CoversIndian, Fine-tuned CQTNet)

| Method                       | MAP    | Top10  | Rank1 |
|------------------------------|--------|--------|-------|
| DA + Source Audio            | 0.7901 | 0.0899 | 5     |
| DA + VocalSep Audio          | 0.9234 | 0.0975 | 2     |

> Modified Triplet Loss outperforms standard Triplet Loss on both source audio and vocal-separated data.

---

### 3.6 Consolidated Results on CoversIndian Test Set (Fine-tuned CQTNet Base)

| Method                           | MAP    | Top10  | Rank1 |
|----------------------------------|--------|--------|-------|
| VocalSep                         | 0.9001 | 0.0962 | 2     |
| DA + VocalSep                    | 0.9131 | 0.0958 | 2     |
| Trip.Loss + VocalSep             | 0.8839 | 0.0979 | 3     |
| Trip.Loss + DA + VocalSep        | 0.8953 | 0.0971 | 3     |
| Mod Trip.Loss + VocalSep         | 0.8922 | 0.0962 | 2     |
| **Mod Trip.Loss + DA + VocalSep**| **0.9234** | **0.0975** | **2** |

---

### 3.7 CQTNet + MERT / HuBERT (CoversIndian)

| Model               | MAP    | Top10  | Rank1 |
|---------------------|--------|--------|-------|
| CQTNet + MERT       | 0.5017 | 0.0664 | 28    |
| CQTNet + HuBERT     | 0.4466 | 0.0601 | 37    |

> Fusing large audio model embeddings with CQT features yielded **no improvement**. MERT/HuBERT weights were kept frozen; trainable fusion or attention-based merging may be explored in future work.
