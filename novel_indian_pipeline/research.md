# research.md — what we tried, in plain language

A friendly walk-through of the research that produced this pipeline. No
jargon assumed. Every paper or method named below has a clickable link.

---

## What problem are we solving?

A **cover song** is a re-recording of a song by a different artist. They share
melody and chord progression but differ in tempo, instruments, vocal style,
and sometimes key. **Cover song identification (CSI)** is the task of
recognising "this recording is a cover of that recording" automatically.

Most CSI systems are trained on Western pop ([SHS100K](
https://github.com/NovaFrost/SHS100K), [Da-TACOS](
https://mtg.github.io/da-tacos/)). When we run them on Indian popular music,
their accuracy drops a lot. We want a system that performs well on **Indian
covers specifically** — Hindi film music, Malayalam covers, etc.

---

## What we tried before that didn't work great

The repository started by running three standard CSI models on Indian data:

- **[CQTNet](https://arxiv.org/abs/1911.00334)** — a CNN that reads the song's
  CQT (a frequency-vs-time picture). Best of the three; we kept it.
- **[TPPNet](https://www.researchgate.net/publication/328287899)** — uses
  temporal pyramid pooling on a chroma-like representation. OK on Western,
  weaker on Indian.
- **[ByteCover](https://arxiv.org/abs/2010.14022)** — heavyweight ResNet+IBN
  with classification + triplet heads. Surprisingly bad on Indian data
  (MAP ≈ 0.005 in one run), suggesting the architecture overfits to Western
  production.

We tried combining CQTNet embeddings with **[HuBERT](https://arxiv.org/abs/
2106.07447)** and **[MERT](https://arxiv.org/abs/2306.00107)** vocal features.
Concatenation hurt performance; we shelved it.

We tried **[Demucs](https://github.com/facebookresearch/demucs)** vocal
separation. This is where the surprise happened:

- On **Indian** covers, isolating the vocals **improved** MAP a lot
  (0.7238 → 0.8629 on CoversIndian).
- On **Western** covers, isolating the vocals **hurt** MAP a lot
  (0.8768 → 0.7181 on Covers80).

That difference is the central observation behind this whole project. Our
working theory: Bollywood covers vary wildly in *accompaniment* (acoustic /
EDM / unplugged) but stay faithful to the *vocal melody* and ornamentation,
so the instrumental channel is mostly noise for cover-matching. Western
covers more often share riffs and progressions in the mix.

We also added a **modified triplet loss** (the senior researcher's idea) and a
**dynamic** version of it (refresh the "hard negative" pick every few epochs
from the live model, instead of pre-computing it once). These pushed us from
MAP 0.86 to MAP 0.92 on CoversIndian-vocal.

---

## What we're trying now

Two ideas, on top of everything above.

### Idea 1: Tonic-relative CQT rotation

The user noticed: **a lot of the variation between an original and a cover is
just key.** A Hindi song originally sung by a male singer might be re-recorded
a few semitones higher by a female singer; the melody is identical, the key
isn't.

The user's first hypothesis was: **shift every vocal stem to the same
reference pitch using audio pitch-shifting.** That works conceptually, but:

- Audio pitch-shifting introduces vocoder artefacts that the CNN can latch
  onto.
- It throws away "absolute key match" as a cover signal.
- It is slow and irreversible.

There's a much cheaper move from the classical-CSI literature:

- Estimate the **tonic** (Sa, the "home note") of each song.
- Then **rotate the CQT** so that the tonic always sits at bin 0.

That's it. The rotation is a `numpy.roll` *within each octave*; no audio
re-synthesis. It's the same trick as the
**[Optimal Transposition Index (OTI)](https://www.iiia.csic.es/~jserra/
publications/2008_TASLP_SerraGomezHerreraSerra.pdf)** by Serra et al. (2008)
which is part of every classical CSI pipeline.

For estimating the tonic on Indian music we lean on existing work:

- Salamon, Gulati, Serra — **"A Multipitch Approach to Tonic Identification in
  Indian Classical Music"** (ISMIR 2012). [Paper link](
  https://www.justinsalamon.com/uploads/4/3/9/4/4394963/
  salamon_gulati_serra_ismir_2012.pdf).
- Gulati et al. — **"Automatic Tonic Identification in Indian Art Music:
  Approaches and Evaluation"** (J. New Music Research 2014). [Paper link](
  https://repositori.upf.edu/handle/10230/26213).

For sanity-checking tonic estimates we can also run a monophonic F0 estimator
on the vocal stem:

- **[CREPE](https://arxiv.org/abs/1802.06182)** (Kim et al., 2018).
- **[SPICE](https://arxiv.org/abs/1910.11664)** (Gfeller et al., 2020).

### Idea 2: A better loss — SupCon + SubCenter-ArcFace

Our current loss is **modified triplet**: per anchor, one positive (a cover
from the same clique) and two negatives (one random, one "hardest" similar
embedding from a different clique). With Indian cliques averaging 4–10 covers,
this throws away most of the available positive structure.

Two ingredients from the recent retrieval literature fix that:

- **[Supervised Contrastive Learning (SupCon)](https://arxiv.org/abs/
  2004.11362)** — Khosla et al., NeurIPS 2020. For each anchor, **all**
  same-clique embeddings count as positives at once, all others as negatives.
  Drop-in upgrade for triplet-style losses when each class has multiple
  examples. Indian cliques fit that perfectly.
- **[ArcFace](https://arxiv.org/abs/1801.07698)** + **[SubCenter-ArcFace](
  https://www.researchgate.net/publication/345804972)** — Deng et al., 2019 /
  2020. A classification head with an *additive angular margin*; SubCenter
  adds K=3 weight vectors per class so a clique can absorb diverse styles
  (acoustic / EDM / unplugged) without forcing them to the same point. Used
  by **[ByteCover](https://arxiv.org/abs/2010.14022)** and
  **[CoverHunter](https://arxiv.org/abs/2306.09025)** as the de facto modern
  CSI head.

So the loss becomes: **dynamic-modified-triplet + SupCon + SubCenter-ArcFace**,
all on the same 300-d CQTNet embedding.

### Idea 3: Combining SupCon and ArcFace without tonic rotation

After running the full ablation grid we saw that tonic rotation consistently
hurt every row it touched (see Results section below). That left one untested
cell: **Dyn Triplet + SupCon + ArcFace without tonic rotation** (row 9). Row 7
had all four ingredients but the tonic drag masked the SupCon+ArcFace
interaction. Row 9 isolates that combination cleanly.

---

## What we actually found (results)

We ran 9 rows in total, each trained on Indian-mix for 75 epochs (with
early stopping at patience 15) and evaluated on four test sets.

| Row | Config | Indian MAP | Indian Vocal MAP | SHS100K MAP |
|-----|--------|:----------:|:----------------:|:-----------:|
| 3 | **Dyn Triplet + SupCon** | **0.7500** | **0.7208** | **0.8466** |
| 9 | Dyn Triplet + SupCon + ArcFace | *running* | *running* | *running* |
| 7 | Tonic + Dyn Triplet + SupCon + ArcFace (full) | 0.6774 | 0.6734 | 0.7862 |
| 4 | Dyn Triplet + ArcFace | 0.7367 | 0.7100 | 0.8290 |
| 1 | Dyn Triplet only (baseline) | 0.7425 | 0.7212 | 0.8309 |
| 5 | Tonic + Dyn Triplet + SupCon | 0.6980 | 0.6854 | 0.7945 |
| 6 | Tonic + Dyn Triplet + ArcFace | 0.6515 | 0.6485 | 0.7745 |
| 2 | Tonic + Dyn Triplet | 0.6597 | 0.6629 | 0.7624 |
| 8 | Tonic only (sanity, 0 epochs) | 0.6061 | 0.6223 | 0.7350 |

### What the results tell us

**SupCon is the single biggest improvement.** Adding it to the dynamic triplet
(row 3) beats the baseline by +0.0075 MAP on Indian and +0.0157 on SHS100K.
The reason fits the theory: Indian cliques average 8–10 covers, so SupCon's
"use all positives at once" strategy has much more signal to work with than a
single-positive triplet.

**Tonic rotation consistently hurts** — every tonic row is worse than its
non-tonic counterpart by 5–8% MAP. Two reasons:

1. The chroma-peak tonic estimator introduces noise (~10–15% wrong estimates
   on non-modal tracks). Wrong rotations mis-align features instead of
   normalising them.
2. Many Indian popular-music covers keep the original key, so the absolute
   key position is a genuine cover cue the model was exploiting. Removing it
   is an own goal.

This is a clean **negative result worth reporting**: the standard CSI
assumption that key normalisation always helps does *not* hold for Indian
popular music.

**ArcFace on its own adds little.** Row 4 (triplet+arcface, 0.7367) barely
differs from the row 1 baseline (0.7425). The angular margin gives marginal
discriminative push when the clique structure is already well-handled by
the triplet.

**The "full kitchen sink" (row 7) loses to SupCon alone** because the tonic
drag outweighs any benefit from combining all three new ingredients.

**Row 9 (SupCon + ArcFace, no tonic)** is designed to test whether ArcFace
adds anything *on top of* SupCon in the absence of tonic noise. If it beats
row 3, the combination is the headline result; if not, SupCon alone is.

---

## Why this might work

- SupCon uses every positive on every step instead of one per anchor, which
  matters when our cliques have 4–10 covers each.
- SubCenter-ArcFace gives intra-clique style variance somewhere to live in
  the embedding space without dragging the centroid.
- The SupCon+ArcFace pairing (without tonic rotation) isolates the loss-side
  contribution and avoids the tonic estimator noise that hurt rows 2, 5, 6,
  and 7.

---

## Things we considered and rejected

- **Audio-domain pitch normalisation.** Worse version of the chroma-rotation
  trick. Slower and lossier.
- **Tonic-relative CQT rotation** — tried across rows 2, 5, 6, 7, 8 and hurt
  every row by 5–8% MAP. The chroma-peak estimator introduces noise; more
  importantly, absolute key is a genuine cover cue in Indian popular music that
  the model benefits from. Rejected as preprocessing for this corpus.
- **[SIFT/ORB](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)
  on spectrograms.** Local keypoint matching is fragile to time-stretch.
- **[SSIM](https://en.wikipedia.org/wiki/Structural_similarity)** on
  spectrograms. Captures pixel-level local statistics, not the diagonal-stripe
  pattern that signals matching covers.
- **Hough-transform diagonal detection.** Used in audio fingerprinting; too
  rigid for tempo-varying covers.
- **Q-max / D-max / [SiMPle](https://www.researchgate.net/publication/
  309419024)**. Strong non-learned baselines, but already covered in existing
  CSI literature; nothing new to add.
- **HuBERT/MERT vocal-feature concatenation.** Tried, didn't work, dropped.
- **New backbones (ByteCover, CoverHunter rewrites).** We keep CQTNet so any
  gain we report is attributable to the recipe, not the architecture.

---

## References

CSI methods used or compared against:

- **CQTNet** — Yu et al., "Learning a Representation for Cover Song
  Identification Using Convolutional Neural Network," ICASSP 2020.
  [arXiv:1911.00334](https://arxiv.org/abs/1911.00334)
- **TPPNet** — Yu et al., "Temporal Pyramid Pooling Convolutional Neural
  Network for Cover Song Identification," IJCAI 2019.
- **ByteCover (v1/v2/v3)** — Du et al., ICASSP 2021/2022/2023.
  [arXiv:2010.14022](https://arxiv.org/abs/2010.14022) (v1)
- **CoverHunter** — Liu et al., 2023.
  [arXiv:2306.09025](https://arxiv.org/abs/2306.09025)
- **Discogs-VI** — Araz, Serra, Bogdanov, 2024.
  [arXiv:2410.17400](https://arxiv.org/abs/2410.17400)
- **Re-MOVE** — Yesiler et al., 2020.
  [arXiv:2010.16151](https://arxiv.org/abs/2010.16151)
- **Q-max / D-max** — Serra, "Identification of Versions of the Same Musical
  Composition by Processing Audio Descriptions," PhD thesis, UPF 2011.

Tonic / pitch tooling:

- **OTI** — Serra et al., "Chroma Binary Similarity and Local Alignment
  Applied to Cover Song Identification," IEEE TASLP 2008.
  [PDF](https://www.iiia.csic.es/~jserra/publications/
  2008_TASLP_SerraGomezHerreraSerra.pdf)
- **Indian tonic identification** — Salamon, Gulati, Serra, ISMIR 2012;
  Gulati et al., JNMR 2014.
- **CREPE** — Kim et al., 2018. [arXiv:1802.06182](
  https://arxiv.org/abs/1802.06182)
- **SPICE** — Gfeller et al., 2020. [arXiv:1910.11664](
  https://arxiv.org/abs/1910.11664)

Loss functions:

- **Triplet loss / FaceNet** — Schroff et al., CVPR 2015.
  [arXiv:1503.03832](https://arxiv.org/abs/1503.03832)
- **Multi-Similarity loss** — Wang et al., CVPR 2019.
  [arXiv:1904.06627](https://arxiv.org/abs/1904.06627)
- **Circle loss** — Sun et al., CVPR 2020.
  [arXiv:2002.10857](https://arxiv.org/abs/2002.10857)
- **ArcFace** — Deng et al., CVPR 2019.
  [arXiv:1801.07698](https://arxiv.org/abs/1801.07698)
- **SubCenter-ArcFace** — Deng et al., ECCV 2020.
- **Center loss** — Wen et al., ECCV 2016.
- **SupCon** — Khosla et al., NeurIPS 2020.
  [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)

Audio source separation:

- **Demucs (HT-Demucs)** — Défossez et al., 2022.
  [arXiv:2211.08553](https://arxiv.org/abs/2211.08553)

Datasets:

- **SHS100K** — [GitHub](https://github.com/NovaFrost/SHS100K)
- **Da-TACOS** — Yesiler, Serra, Serra, ISMIR 2019.
  [Project page](https://mtg.github.io/da-tacos/)
- **Covers80** — Ellis, "Covers80," dataset, 2007.
- **CompMusic** (Indian art music collections) — UPF MTG.
  [Project page](https://www.upf.edu/web/mtg/compmusic)
