
# Results 

## CQTNet 

### SHS100K-test

CQTNet trained on SHS100K-train using cross-entropy loss, evaluated on full audio and Demucs vocal-separated test splits.

| Dataset           | MAP        | Top10      | Rank1 (MRR) |
| ----------------- | ---------- | ---------- | ----------- |
| SHS100K **      | 0.8235 | 0.8656 | 6.87    |
| SHS100K Vocal | 0.3926 | 0.5377 | 24.07   |

### Indian Covers

SHS100K-pretrained CQTNet (Baseline) and a version further fine-tuned on Indian Covers with cross-entropy loss (Finetuned), evaluated on Indian Covers source audio and Demucs vocal-separated test sets.

| Dataset                  | Model            | MAP    | Top10  | Rank1 |
| ------------------------ | ---------------- | ------ | ------ | ----- |
| Indian Covers test       | Baseline CQTNet  | 0.7366 | 0.3811 | 6.42  |
| Indian Covers test       | Finetuned CQTNet | 0.7366 | 0.3811 | 6.42  |
| Indian Covers Vocal test | Baseline CQTNet  | 0.7113 | 0.3547 | 8.21  |
| Indian Covers Vocal test | Fintuned CQTNet  | 0.6811 | 0.3457 | 10.72 |



## TPPNET

### SHS100K-test

TPPNet (Temporal Pyramid Pooling Network) trained on SHS100K-train with cross-entropy loss, evaluated on full audio and vocal-separated test splits.

| Dataset           | MAP    | Top10  | Rank1 (MRR) |
| ----------------- | ------ | ------ | ----------- |
| **SHS100K **      | 0.5672 | 0.6860 | 9.41        |
| **SHS100K Vocal** | 0.1874 | 0.3058 | 34.96       |

### Indian Covers

TPPNet pretrained on SHS100K used as baseline, and fine-tuned on Indian Covers with cross-entropy loss (Finetuned), evaluated on Indian Covers source audio and vocal-separated test sets.

| Dataset                  | Model            | MAP    | Top10   | Rank1  |
| ------------------------ | ---------------- | ------ | ------- | ------ |
| Indian Covers test       | Baseline CQTNet  | 0.4514 |  0.2406 | 20.44  |
| Indian Covers test       | Finetuned CQTNet | 0.5002 | 0.2726  | 18.14  |
| Indian Covers Vocal test | Baseline CQTNet  | 0.5506 | 0.2911  | 14.48  |
| Indian Covers Vocal test | Fintuned CQTNet  | 0.5368 | 0.2814  |  15.58 |

## ByteCover

### SHS100K-test

ByteCover (ResNet50-based architecture with GeM pooling) trained on SHS100K using combined triplet and cross-entropy loss, evaluated on full audio and vocal-separated test splits.

| Dataset           | MAP    | Top10  | 
| ----------------- | ------ | ------ |
| SHS100K       | 0.2819 | 0.8246 |
| SHS100K Vocal | 0.1154 | 0.6504 | 

### Indian Covers

SHS100K-pretrained ByteCover model evaluated zero-shot on Indian Covers source audio and vocal-separated test sets (no fine-tuning).

| Dataset                  | Model            | MAP    | Top10   | 
| ------------------------ | ---------------- | ------ | ------- | 
| Indian Covers test       | Baseline CQTNet  | 0.2934 | 0.6615 | 
| Indian Covers Vocal test | Baseline CQTNet  | 0.2889 | 0.7094  | 



## Triplet Loss 

CQTNet fine-tuned from the cross-entropy checkpoint using standard triplet loss (margin=0.05) with data augmentation (pitch shift ±4 semitones, tempo stretch ×0.8/1.2), evaluated on Indian Covers and SHS100K test sets.

| Dataset             | Model                  | MAP        | Top10      | Rank1    |
| ------------------- | ---------------------- | ---------- | ---------- | -------- |
| Indian Covers       | Triplet fine-tuned     | 0.7460     | 0.3825     | **4.40** |
| Indian Covers Vocal | Triplet fine-tuned     | 0.7279     | 0.3628     | 9.68     |
| ~~SHS100K-Test~~    | ~~Triplet fine-tuned~~ | ~~0.8871~~ | ~~0.9090~~ | ~~1.41~~ |
| SHS100K-Test Vocal  | Triplet fine-tuned     | 0.5950     | 0.7255     | 4.20     |



## Dynamic Modified Triplet Loss

CQTNet fine-tuned using a modified triplet loss (margin=0.3) that mines the hardest in-batch negative alongside a random negative, with the same data augmentation pipeline, targeting improved discrimination for Indian music.

| Dataset             | Model            | MAP    | Top10  | Rank1 |
| ------------------- | ---------------- | ------ | ------ | ----- |
| Indian Covers       | Modified triplet | 0.7620 | 0.5566 | 4.53  |
| Indian Covers Vocal | Modified triplet | 0.8043 | 0.5836 | 3.37  |
| ~~SHS100K-Test~~        | ~~Modified triplet~~ | ~~0.8839~~ | ~~0.9059~~ | ~~1.38~~  |
| SHS100K-Test Vocal  | Modified triplet | 0.6717 | 0.7509 | 4.14  |
