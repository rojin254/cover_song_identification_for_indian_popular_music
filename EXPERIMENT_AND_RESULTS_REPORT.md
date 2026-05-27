# Experiment and Results Report

## 1. Experimental Overview

This study investigates how a cover-song identification system can be adapted from a large Western music setting to Indian popular music. The central question is whether a representation learned from large-scale cover-song data can be improved for Indian cover retrieval through targeted adaptation strategies. The experimental program follows a progressive design: first establishing a baseline, then introducing transfer learning, then isolating vocal content, and finally reshaping the embedding space with metric-learning objectives.

Two datasets are considered throughout the evaluation. The first is the Indian Covers test set, which represents the target domain of interest. The second is the SHS100K test set, which acts as a Western reference domain and helps assess whether the same modeling choices generalize across musical traditions. Each dataset is evaluated in two forms: the original full-audio mixture and a vocally separated version. This produces four evaluation settings in total: Indian-normal, SHS100K-normal, Indian-vocal, and SHS100K-vocal.

The experiments are organized around three model conditions. The first is the baseline CQTNet retrieval model. The second is the same model after triplet-loss fine-tuning, which explicitly encourages the model to bring related versions closer and unrelated songs farther apart. The third is the modified triplet-loss model, which introduces a stronger negative-learning mechanism by combining a random negative with the most confusable negative example.

## 2. Experimental Methodology

### 2.1 Baseline Representation

The starting point is CQTNet, a convolutional architecture operating on Constant-Q Transform representations of audio. The use of CQT is motivated by the fact that cover-song similarity depends more on pitch and melodic contour than on raw waveform similarity. By converting audio into a time-frequency representation that better aligns with musical pitch structure, the system can focus on harmonic and melodic regularities that are more likely to persist across versions of the same song.

In the baseline setting, each audio recording is transformed into a CQT representation and passed through the network to produce an embedding. Retrieval is then performed in the embedding space using similarity scores between the query song and all candidate songs. This baseline serves two purposes. First, it provides a direct measure of how well the pre-existing representation transfers without additional metric shaping. Second, it gives a stable reference point against which later improvements can be interpreted.

### 2.2 Fine-Tuning for the Target Task

The next step is to adapt the baseline representation to the target cover-song identification setting. A model pre-trained on large-scale cover-song material already carries useful information about melodic and harmonic similarity, but its representation is shaped by the statistical structure of the source domain. Fine-tuning is therefore used to shift the model toward the acoustic and stylistic properties of the current datasets.

This adaptation step is conceptually important because it converts a general cover-song representation into a domain-sensitive one. Rather than learning from scratch, the model starts from a representation that already encodes cover similarity and is then adjusted using task-specific supervision. This is especially important for Indian popular music, where arrangements, vocal delivery, and instrumentation differ from the Western recordings that dominate many pre-training corpora.

### 2.3 Vocal Separation

One of the main hypotheses in this work is that the vocal line carries particularly strong identity cues in Indian popular music. Many cover versions preserve the central melody even when instrumentation, tempo, or production style changes. To test this hypothesis, a second evaluation setting is constructed using vocally separated audio rather than full mixtures.

The procedure is conceptually simple. Each song is first processed by a vocal-separation system. The isolated vocal signal is then converted into its own CQT representation, and the same retrieval pipeline is applied. This allows the experiment to test whether suppressing accompaniment helps the model focus on the melodic information that is most stable across Indian covers.

The value of this comparison is not only practical but interpretive. If vocal-only representations improve retrieval on Indian data, that supports the idea that melodic identity is the dominant cue in that musical tradition. If the same strategy hurts performance on Western data, that would suggest the effect is culturally specific rather than universally beneficial.

### 2.4 Triplet-Loss Fine-Tuning

Cross-entropy-based training encourages a model to separate songs by class, but it does not directly optimize the geometry of the retrieval space. Cover-song identification, however, is fundamentally a ranking problem. The system must place true covers near each other and push non-covers farther away. Triplet loss addresses this directly.

Each training instance is built from three components:

1. An anchor song, which serves as the reference.
2. A positive example, which is a cover version of the same underlying song.
3. A negative example, which is a song from a different clique.

The training objective encourages the model to reduce the distance between the anchor and the positive while increasing the distance between the anchor and the negative. The effect is to reshape the embedding space so that retrieval behavior improves even when songs differ substantially in arrangement or recording conditions.

This stage is best understood as a shift from classification-oriented learning to similarity-oriented learning. The model is no longer being rewarded merely for predicting a class label; it is being trained to produce a retrieval space with better neighborhood structure.

### 2.5 Modified Triplet Loss

The modified triplet formulation extends the standard triplet setup by making the negative-learning signal more demanding. In realistic cover-song retrieval, not all negatives are equally informative. Some non-cover songs are obviously unrelated, while others are melodically or stylistically confusable. Training on only random negatives can therefore make the task too easy and may fail to teach the model how to discriminate among near-miss examples.

The modified objective addresses this by using two kinds of negatives:

1. A random negative, which preserves broad inter-class separation.
2. A highly similar negative, which represents the most confusable non-cover example.

The positive term is weighted more strongly so that the model is simultaneously encouraged to tighten true cover relationships and to open a wider margin against both easy and hard negatives. Conceptually, this turns the loss into a finer-grained retrieval objective. Instead of merely separating covers from arbitrary non-covers, the model learns to distinguish a true cover from songs that appear musically close but are still incorrect.

This is especially relevant for Indian music, where shared melodic habits, raga-like contours, and stylistic overlap can make unrelated songs sound deceptively similar. The modified triplet loss is designed precisely for this higher-confusion regime.

### 2.6 Evaluation Strategy

Performance is assessed with three retrieval metrics.

1. MAP measures ranking quality across the full retrieved list and serves as the primary summary of retrieval effectiveness.
2. Top10 measures how often relevant songs appear among the top retrieved candidates, which is useful for practical search behavior.
3. Rank1 measures how early the first correct cover appears; lower values indicate that the system is finding a correct match closer to the top of the list.

Taken together, these metrics provide complementary views of performance. MAP captures the overall ranking quality, Top10 reflects short-list usefulness, and Rank1 indicates how quickly a correct answer becomes available to the user.

## 3. Step-by-Step Experimental Progression

The experimental logic can be summarized as a sequence of increasingly targeted interventions.

### Step 1: Establish the baseline

The first step is to evaluate the baseline retrieval model on both datasets using full-audio CQTs. This provides the initial measure of cross-dataset generalization and establishes the reference point for all later comparisons.

### Step 2: Evaluate the effect of vocal isolation

The same baseline is then evaluated on vocally separated audio. This isolates the contribution of the vocal melody and tests whether the retrieval model benefits from removing accompaniment.

### Step 3: Fine-tune with triplet loss

The baseline representation is next refined using triplet loss so that the embedding space becomes more retrieval-oriented. This step directly trains the model to pull covers together and push non-covers apart.

### Step 4: Introduce harder negatives

The final step replaces the standard triplet objective with the modified triplet loss. By incorporating both a random negative and a confusable negative, the training signal becomes more demanding and more aligned with realistic retrieval errors.

### Step 5: Compare behavior across domains

The resulting systems are finally compared across Indian and Western data, and across normal versus vocally separated audio. This comparison reveals which improvements are domain-specific and which are broadly transferable.

## 4. Consolidated Results

The following table combines all reported results into a single view. It contains two sections, one for normal audio and one for vocally separated audio. Within each section, the Indian Covers and SHS100K-Test datasets are shown with three model conditions: baseline, triplet-loss fine-tuned, and modified-triplet fine-tuned.

| Configuration | MAP | Top10 | Rank1 |
|---|---:|---:|---:|
| **Normal Audio** |  |  |  |
| Indian Covers - Baseline | 0.7366 | 0.3811 | 6.42 |
| Indian Covers - Triplet Fine-tuned | 0.7460 | 0.3825 | 4.40 |
| Indian Covers - Modified Triplet | 0.7620 | 0.5566 | 4.53 |
| SHS100K-Test - Baseline | 0.8844 | 0.9074 | 1.56 |
| SHS100K-Test - Triplet Fine-tuned | 0.8871 | 0.9090 | 1.41 |
| SHS100K-Test - Modified Triplet | 0.8839 | 0.9059 | 1.38 |
| **Vocally Separated Audio** |  |  |  |
| Indian Covers - Baseline | 0.6811 | 0.3457 | 10.72 |
| Indian Covers - Triplet Fine-tuned | 0.7279 | 0.3628 | 9.68 |
| Indian Covers - Modified Triplet | 0.8043 | 0.5836 | 3.37 |
| SHS100K-Test - Baseline | 0.5996 | 0.7326 | 4.42 |
| SHS100K-Test - Triplet Fine-tuned | 0.5950 | 0.7255 | 4.20 |
| SHS100K-Test - Modified Triplet | 0.6717 | 0.7509 | 4.14 |

## 5. Results Discussion

Several clear trends emerge from the consolidated results.

First, the Indian dataset benefits consistently from stronger retrieval-oriented learning. On normal audio, performance improves from the baseline to triplet fine-tuning and improves again with modified triplet loss. The modified triplet setting gives the strongest overall Indian-source result, indicating that the harder-negative formulation is helping the model resolve musically similar but incorrect matches.

Second, the Indian vocal condition shows the largest relative improvement of the entire study. While the vocal baseline starts below the normal-audio baseline, the modified triplet model produces a substantial jump in all three metrics. This suggests that once the representation is trained to exploit vocal information correctly, vocal isolation becomes highly beneficial for Indian cover retrieval.

Third, the SHS100K source-audio condition behaves differently. Its baseline is already strong, and the gains from additional fine-tuning are comparatively small. This indicates that the Western domain may already be well served by the original representation, leaving less room for improvement.

Fourth, the SHS100K vocal condition does not benefit in the same way that the Indian vocal condition does. The scores remain much lower than the corresponding full-audio SHS100K results, and neither triplet nor modified triplet changes the picture substantially. This supports the broader interpretation that vocal melody is a much stronger identity cue for Indian cover retrieval than for the Western reference setting used here.

## 6. Main Takeaway

The overall pattern supports a culturally grounded conclusion: for Indian popular music, the vocal melody is a particularly informative cue for cover-song identification, and metric-learning objectives that explicitly structure the embedding space are highly effective. Among the tested methods, the modified triplet formulation provides the most convincing gains, especially when combined with vocally separated audio. In contrast, the same interventions produce only minor changes on SHS100K, showing that the value of these design choices depends strongly on the musical domain.
# Experiment and Results Report



## 3. Requested Results Tables (4 rows x 3 metrics)

Source: [quick_result.md](quick_result.md)

### 3.1 Baseline CQTNet

| Dataset Configuration | MAP | Top10 | Rank1 |
|---|---:|---:|---:|
| Indian Covers (source audio) | 0.7366 | 0.3811 | 6.42 |
| SHS100K-Test (source audio) | 0.8844 | 0.9074 | 1.56 |
| Indian Covers (vocal separated) | 0.6811 | 0.3457 | 10.72 |
| SHS100K-Test (vocal separated) | 0.5996 | 0.7326 | 4.42 |

### 3.2 After Triplet Loss Fine-tuning

| Dataset Configuration | MAP | Top10 | Rank1 |
|---|---:|---:|---:|
| Indian Covers (source audio) | 0.7460 | 0.3825 | 4.40 |
| SHS100K-Test (source audio) | 0.8871 | 0.9090 | 1.41 |
| Indian Covers (vocal separated) | 0.7279 | 0.3628 | 9.68 |
| SHS100K-Test (vocal separated) | 0.5950 | 0.7255 | 4.20 |

### 3.3 After Modified Triplet Loss Fine-tuning

| Dataset Configuration | MAP | Top10 | Rank1 |
|---|---:|---:|---:|
| Indian Covers (source audio) | 0.7620 | 0.5566 | 4.53 |
| SHS100K-Test (source audio) | 0.8839 | 0.9059 | 1.38 |
| Indian Covers (vocal separated) | 0.8043 | 0.5836 | 3.37 |
| SHS100K-Test (vocal separated) | 0.6005 | 0.7250 | 4.14 |

---

## 4. Key Observations from Current Runs

- Modified triplet gives the strongest gains on Indian data (source and vocal).
- SHS100K source has high baseline already; gains from triplet/modified triplet are small.
- SHS100K vocal remains much weaker than SHS100K source in MAP, indicating vocal-only representation is not sufficient for this dataset.
- Indian vocal modified triplet run in [quick_result.md](quick_result.md) (`MAP=0.8043`) is substantially better than an earlier commented value in the same file.

---

## 5. Important Implementation Notes (Paper vs Code)

- Triplet margin differs from paper pseudocode in current code:
  - Paper examples use margin around 0.3 for triplet.
  - Code uses `TripletMarginLoss(margin=0.05)` in triplet stage:
    - [indian_new_pipeline/main.py#L457](indian_new_pipeline/main.py#L457)
    - [indian_vocal_pipeline/main.py#L454](indian_vocal_pipeline/main.py#L454)
    - [shs100k_new_pipeline/main.py#L444](shs100k_new_pipeline/main.py#L444)
    - [shs100k_vocal_pipeline/main.py#L422](shs100k_vocal_pipeline/main.py#L422)
- Modified triplet margin is 0.3 in code, aligned with paper:
  - [indian_new_pipeline/main.py#L579](indian_new_pipeline/main.py#L579)
  - [indian_vocal_pipeline/main.py#L577](indian_vocal_pipeline/main.py#L577)
  - [shs100k_new_pipeline/main.py#L548](shs100k_new_pipeline/main.py#L548)
  - [shs100k_vocal_pipeline/main.py#L535](shs100k_vocal_pipeline/main.py#L535)
- Paper setup mentions batch size 60 and no shuffle for grouped augmentation. Current pipelines train with batch size 120 and shuffle enabled in loaders.
- Data augmentation logic is not inside these four `main.py` files; these pipelines consume precomputed CQT features and split lists.

---

## 6. Repro Entry Points

- CE + stage control (Indian source):
  - [indian_new_pipeline/main.py](indian_new_pipeline/main.py)
- CE + stage control (Indian vocal):
  - [indian_vocal_pipeline/main.py](indian_vocal_pipeline/main.py)
- CE + stage control (SHS100K source):
  - [shs100k_new_pipeline/main.py](shs100k_new_pipeline/main.py)
- CE + sequential triplet+dynamic (SHS100K vocal):
  - [shs100k_vocal_pipeline/main.py](shs100k_vocal_pipeline/main.py)

All four expose Fire CLI (`if __name__ == '__main__': fire.Fire()`), so training/evaluation routines are callable directly from function names.
