# NOVEL_INDIAN_CSI_PROPOSAL.md

A novel preprocessing + training recipe for cover song identification on
Indian popular music.

---

## 1. Context and motivation

Cover song identification (CSI) systems trained on Western corpora (SHS100K,
Da-TACOS, Discogs-VI) generalise poorly to Indian popular music. Prior work on
this codebase (see [Final_Combined_Result.md](../Final_Combined_Result.md),
[ISMIR_CSI_PAPER_CONTENT.md](../ISMIR_CSI_PAPER_CONTENT.md),
[CSI_WORK_DONE.md](../CSI_WORK_DONE.md), and
[interim_report.tex](../interim_report.tex)) established three findings:

1. **CQTNet** is the strongest backbone we have, beating **TPPNet** and
   **ByteCover** on both Western and Indian sets after fine-tuning.
2. **Vocal separation (Demucs) helps Indian covers and hurts Western covers.**
   On *CoversIndian*, baseline CQTNet jumps from MAP **0.7238 → 0.8629** when
   fed Demucs vocal stems; on *Covers80*, the same intervention drops MAP from
   **0.8768 → 0.7181**.
3. The **modified triplet loss** (Section 1.6 of the paper:
   `L = max(0, α + 2·d(A,P) − [d(A,N_sim) + d(A,N_rand)])`,
   with hardest-in-batch N_sim mining) plus pitch/tempo augmentation produces
   the current best Indian result: **MAP 0.9234** on CoversIndian.
4. The user's **dynamic** variant of modified triplet loss (N_sim refreshed
   every 3 epochs from live model embeddings; see
   [indian_new_pipeline/main.py:677](../indian_new_pipeline/main.py#L677))
   removes the static-lookup dependency but is otherwise a small refinement.

Two angles remain unexplored:

- **Key invariance.** A cover may share melody and harmony with its original
  while sitting in a different key. The current pipeline relies entirely on
  the CNN's learned invariance; no explicit normalisation step exists.
- **Loss expressiveness.** The modified triplet loss uses *one* positive and
  *two* negatives per anchor. With Indian cliques averaging 4–10 covers
  (Hindi 8–10/clique, Malayalam 2–3), this discards most of the available
  positive structure on every step.

This document proposes a four-piece recipe that addresses both angles, and an
ablation matrix that cleanly attributes any gain to each piece.

---

## 2. Hypothesis

> Combining (a) **tonic-relative CQT rotation** as preprocessing with (b) a
> **SupCon + SubCenter-ArcFace + dynamic-modified-triplet** training objective
> produces an Indian-tuned CSI model that improves MAP on Indian-vocal test
> while preserving (or improving) MAP on the SHS100K-vocal evaluation set.

The hypothesis is decomposed into four falsifiable claims, each tested by one
or more rows of the ablation matrix in §6.

---

## 3. The user's pitch-normalisation idea, refined

**Original framing.** "Estimate a mean frequency of songs and normalise all
vocally-separated data to that pitch, so all vocals sound the same way and the
remaining differences carry the cover-vs-not signal."

**Why audio-domain pitch-shift is the wrong shape.** Re-synthesising every
vocal stem to a fixed reference pitch via librosa/pyrubberband would:
- introduce phase-vocoder artefacts that the CNN can latch onto;
- discard the *relative* key relationship that classical CSI exploits
  (Serra et al.'s OTI does the same job more cheaply on chroma);
- preserve melodic contour (intervals are scale-invariant) but smear vocal
  formants, slightly degrading timbre stability.

**Recommended form: tonic-relative CQT rotation.** Estimate the tonic Sa per
song, then circularly rotate the CQT *within each octave* so that bin 0 is the
tonic. The 84-bin CQT in this codebase is exactly 7 octaves × 12 bins per
octave, so within-octave rotation is straightforward. After rotation:

- All songs share a "tonic = bin 0" frame; the CNN no longer has to learn
  global-key invariance, only within-key variation.
- No re-synthesis, no artefacts, ~milliseconds per track.
- One pass over the existing `.npy` CQT cache produces a `tonic_index.json`
  sidecar; training reads the JSON and never re-estimates.

**Tonic estimator.** Two options are exposed behind a flag for ablation:
1. **Chroma argmax (default):** average CQT across time, fold to one octave,
   peak bin = tonic. Cheap, no extra dependencies, robust on popular vocals.
2. **CREPE F0 → modulo-12 histogram:** for monophonic separated vocals, run
   CREPE/SPICE F0 estimation, map to semitones modulo 12, take the histogram
   peak. Used as a sanity check, not the main path.

**Open risk.** Rotation removes the *absolute* key cue, which is also a real
cover signal when a cover keeps the original key. We control for this with
ablation rows that compare rotated and non-rotated CQT under matched losses.

---

## 4. The novel loss/training recipe

The current pipeline uses **CE → triplet → modified-triplet → dynamic-
modified-triplet** as a 4-stage curriculum. We extend the final stage with two
additional components, both well-supported by recent retrieval literature:

### 4.1 SupCon (Khosla et al., NeurIPS 2020)

For each anchor in a batch, **all** same-clique embeddings are positives and
the rest are negatives. Loss:

```
ℓ_i = − (1 / |P(i)|) Σ_{p ∈ P(i)} log [ exp(z_i·z_p / τ) /
                                         Σ_{a ≠ i} exp(z_i·z_a / τ) ]
```

τ = 0.07. Embeddings are L2-normalised before the dot product.

**Why for Indian CSI:** triplet/InfoNCE-style losses with one positive at a
time are wasteful when cliques average 4–10 covers — SupCon uses every
positive on every step.

### 4.2 SubCenter-ArcFace (Deng et al., ECCV 2020)

A classification head with K=3 sub-centers per class and additive angular
margin m=0.3 (scale s=30). For each clique, we keep K weight vectors; logits
are computed as the **max** cosine over sub-centers, then the margin is added
to the target class. Standard CrossEntropyLoss on the resulting logits.

**Why for Indian CSI:** ByteCover and CoverHunter both stack ArcFace alongside
triplet — it is the de facto recipe for retrieval-style CSI in 2024-2026.
SubCenter handles intra-clique style variance (acoustic cover vs EDM remix vs
unplugged) which is high in Bollywood.

### 4.3 Combined loss

```
L_total = w_dyn · L_dyn_modified_triplet
        + w_sup · L_supcon
        + w_arc · L_arcface
```

Default weights: `w_dyn = 1.0, w_sup = 0.5, w_arc = 0.5`. SupCon and ArcFace
are applied to the **same 300-dim CQTNet embedding**; ArcFace adds a separate
classification head at training time only (not used at inference).

The dynamic-modified-triplet refresh path
([indian_new_pipeline/main.py:677](../indian_new_pipeline/main.py#L677))
is preserved unchanged.

---

## 5. Datasets

Per the user's instruction, this work is restricted to:

| Tag | Manifest (.env key) | Use |
|-----|----------------------|-----|
| Indian-mix-train | `INDIAN_TRAIN_TXT` | training |
| Indian-mix-test | `INDIAN_TEST_TXT` | evaluation |
| Indian-vocal-train | `INDIAN_VOCAL_TRAIN_TXT` | training |
| Indian-vocal-test | `INDIAN_VOCAL_TEST_TXT` | evaluation |
| SHS100K-test (combined train+test, mix) | `SHS100K_TEST_TXT` | evaluation only |
| SHS100K-test-vocal (combined train+test, vocals) | `SHS100K_VOCAL_TEST_TXT` | evaluation only |

Covers80 is **not** used. The SHS100K rows confirm that an Indian-tuned recipe
does not regress Western performance.

---

## 6. Ablation matrix (8 rows)

Each row trains one CQTNet model on Indian-mix-train **and** another on
Indian-vocal-train (separately), warm-started from the corresponding
`.env`-configured checkpoint, then evaluates the resulting two models on
**all four** test manifests above.

| Row | Tonic-rotated CQT | Dyn-mod triplet | SupCon | ArcFace | Notes |
|----:|:--:|:--:|:--:|:--:|---|
| 1 | – | ✓ | – | – | Baseline reproduction (existing best) |
| 2 | ✓ | ✓ | – | – | Effect of preprocessing alone |
| 3 | – | ✓ | ✓ | – | Effect of SupCon alone |
| 4 | – | ✓ | – | ✓ | Effect of ArcFace alone |
| 5 | ✓ | ✓ | ✓ | – | Preprocessing + SupCon |
| 6 | ✓ | ✓ | – | ✓ | Preprocessing + ArcFace |
| 7 | ✓ | ✓ | ✓ | ✓ | **Headline: full recipe** |
| 8 | ✓ | – | – | – | Sanity: rotation without dynamic triplet |

For each row we report MAP, Top-10, and MR1 on each of the four test sets,
written to `results/tables/{row_name}/{dataset}.csv` and aggregated into
`results/tables/summary.csv` and `results/ablation_matrix.md`.

---

## 7. Verification protocol

1. **Tonic-estimator sanity** — manually label keys for ~20 known Bollywood
   originals and confirm `estimate_tonic_from_cqt` agrees within ±1 semitone
   on ≥ 90%.
2. **Rotation unit test** — synthesise a CQT, rotate it by k semitones,
   confirm tonic-normalised rotation produces an identical post-rotation
   tensor for every k (i.e., the operator is idempotent in the tonic frame).
3. **Row-1 reproduction** — must reproduce the existing best result
   (MAP ≈ 0.9234 on CoversIndian-vocal) within ±0.005, confirming the
   refactor did not regress the baseline.
4. **Per-row training** — same 150-epoch dynamic-triplet schedule as the
   existing pipeline; log val MAP every epoch.
5. **Hard-query spot-check** — for each row, manually inspect 10 query→top-5
   examples to verify gains are not concentrated on easy queries.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Tonic estimation fails on instrumental-heavy tracks | Use vocal stems as input to the estimator; fall back to chroma argmax of the vocal stem when CREPE confidence is low |
| Rotation destroys absolute-key signal that CSI relies on | Row 8 (rotation alone) and Row 1 (no rotation) bracket this; if Row 8 < Row 1 we know rotation is harmful in isolation |
| ArcFace head doesn't transfer from Western warm-start | Initialised from scratch, only the embedding backbone uses the warm-start; the existing CE→triplet→modtriplet curriculum handles the cold-start head fine in prior runs |
| Combined loss is unstable | Per-row gradient norm logging; downscale `w_sup` and `w_arc` if loss diverges in first 5 epochs |
| OOM with sub-center ArcFace at K=3 over thousands of cliques | K is bounded by clique count; for SHS100K-scale eval, head is unused at inference, so memory pressure is training-only |

---

## 9. Out of scope

- **New backbones** (ByteCover/CoverHunter rewrites). We keep CQTNet so any
  gain is attributable to the recipe, not the architecture.
- **Audio-domain pitch shift.** Replaced by tonic-relative CQT rotation (§3).
- **HuBERT/MERT fusion.** Prior experiments failed (MAP 0.45–0.50); not
  revisited.
- **Dataset growth.** Work on the existing 541-clique Indian corpus.

---

## 10. Deliverables

1. This document — `NOVEL_INDIAN_CSI_PROPOSAL.md`.
2. Layperson-friendly research note with paper links — `research.md`.
3. Tonic preprocessing module — `tonic.py`, `cqt_rotate.py`,
   `tonic_index_build.py`.
4. Generalised CQT loader — `cqt_loader.py`.
5. New loss module — `losses.py`.
6. Training/evaluation entrypoint — `main.py`.
7. Ablation driver — `run_ablations.sh`.
8. Filled-in results — `results/ablation_matrix.md` and per-row CSVs under
   `results/tables/`.

---

## 11. Results (TO BE FILLED)

To be populated by `run_ablations.sh` and `results/build_summary.py`.

| Row | Indian-mix MAP | Indian-vocal MAP | SHS100K MAP | SHS100K-vocal MAP |
|----:|:-:|:-:|:-:|:-:|
| 1 | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD | TBD |
| 6 | TBD | TBD | TBD | TBD |
| 7 | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD |
