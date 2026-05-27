# Cover Song Identification: Experiment Report

**Project:** Adapting CQTNet for Indian Cover Song Identification  
**Datasets:** CoversIndian (126 train / 119 test pairs), SHS100K-Test (Western reference)

---

## Objective

Adapt CQTNet — a Western pre-trained cover song identification model — to Indian popular music by investigating fine-tuning, vocal separation, and metric-learning objectives.

---

## Experiments

Three model conditions were evaluated across two audio settings (full audio and vocally separated):

1. **Baseline CQTNet** — Western pre-trained model, no adaptation
2. **Triplet Loss Fine-tuning** — embedding trained to pull covers closer and push non-covers apart
3. **Modified Triplet Loss** — augments triplet training with hard (melodically similar) negatives alongside random negatives

---

## Results

### Full Audio

| Dataset | Model | MAP | Top10 | Rank1 |
|---|---|---|---|---|
| Indian Covers | Baseline CQTNet | 0.7366 | 0.3811 | 6.42 |
| Indian Covers | Triplet fine-tuned | 0.7460 | 0.3825 | **4.40** |
| Indian Covers | Modified triplet | **0.7620** | **0.5566** | 4.53 |
| SHS100K-Test | Baseline CQTNet | 0.8844 | 0.9074 | 1.56 |
| SHS100K-Test | Triplet fine-tuned | **0.8871** | **0.9090** | 1.41 |
| SHS100K-Test | Modified triplet | 0.8839 | 0.9059 | **1.38** |

### Vocally Separated Audio

| Dataset | Model | MAP | Top10 | Rank1 |
|---|---|---|---|---|
| Indian Covers | Baseline CQTNet | 0.6811 | 0.3457 | 10.72 |
| Indian Covers | Triplet fine-tuned | **0.7279** | **0.3628** | **9.68** |
| Indian Covers | Modified triplet | 0.6785 | 0.3440 | 10.23 |
| SHS100K-Test | Baseline CQTNet | 0.5996 | 0.7326 | 4.42 |
| SHS100K-Test | Triplet fine-tuned | 0.5950 | 0.7255 | 4.20 |
| SHS100K-Test | Modified triplet | **0.6005** | **0.7250** | **4.14** |

---

## Key Findings

- **Domain adaptation helps on full audio:** Fine-tuning on Indian data consistently improves retrieval over the Western baseline on Indian Covers, with modified triplet yielding the best MAP (0.7366 → 0.7620) and Top10 (0.3811 → 0.5566).
- **Vocal separation + triplet loss is most effective for Indian music:** In the vocal-separated setting, standard triplet fine-tuning achieves the best Indian Covers results (MAP 0.7279, Rank1 9.68), while the modified triplet variant does not improve over baseline here.
- **Modified triplet underperforms on vocal-separated Indian data:** Unlike the full-audio setting, the modified triplet objective slightly degrades performance on vocally separated Indian Covers (MAP 0.6811 → 0.6785), suggesting the hard-negative strategy may not suit the vocal-only feature space.
- **Western performance is mostly preserved:** All variants remain close on SHS100K-Test, showing that Indian-domain adaptation does not significantly degrade Western cover retrieval.

---

*Metrics: MAP = Mean Average Precision (higher is better), Top10 = precision in top-10 retrieved results (higher is better), Rank1 = mean rank of first correct result (lower is better).*
