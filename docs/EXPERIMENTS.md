# Experiment registry and scientific contracts

`csi list`, `csi show --preset NAME`, and [packaged JSON presets](../src/csi/presets)
are the executable registry. There are 5 prerequisites/upstream profiles and
23 report rows. Dataset and modality are independent; every Phase 1 row can
train Indian mix or vocal and defaults to testing all four combinations.

## Independent APIs

| Component | Callable/file |
|---|---|
| Backbone / adapter | `CQTNet.forward`, `ExperimentModel.forward` in [models](../src/csi/models) |
| Tonic | `estimate_tonic`, `rotate_cqt` in [tonic](../src/csi/data/tonic.py) |
| Standard triplet | `triplet_loss` in [triplet](../src/csi/losses/triplet.py) |
| Modified triplet | `modified_triplet_loss`, `modified_from_distances` in [modified_triplet](../src/csi/losses/modified_triplet.py) |
| SupCon | `supcon_loss` in [supcon](../src/csi/losses/supcon.py), returns loss and valid-anchor count |
| SubCenter-ArcFace | `SubCenterArcFace.forward` in [subcenter_arcface](../src/csi/models/subcenter_arcface.py) |
| Random/hard/mean/hard+mean positives | Individual functions/files in [mining](../src/csi/mining), return distance and validity mask |
| Shared nearest negatives | `nearest_negative` in [nsim](../src/csi/mining/nsim.py) |
| Refresh lifecycle | `NegativeCache.refresh`, `state_dict`, `load_state_dict` in [dynamic_nsim](../src/csi/mining/dynamic_nsim.py) |
| Balanced batches | `BalancedSampler`, epoch/state methods in [balanced](../src/csi/sampling/balanced.py) |
| Composition | `Objectives.forward` in [compose](../src/csi/losses/compose.py) |
| Shared engine | `train`, `evaluate`, `preflight` in [training](../src/csi/training.py) |

Modified loss with cosine distance d:

```text
mean(relu(margin + 2*d(A,P) - d(A,N_sim) - d(A,N_rand)))
hard_mean positive term = max(d(A,P_i)) + mean(d(A,P_i))
```

No extra factor two is applied to hard+mean. Mean is mean **distance**, not
distance to a centroid. Hard-only/mean-only use twice the selected distance as
a reconstructed extension of the ordinary formula; their historical
coefficients remain unverified. Loss weight and margin are configurable.
Ordinary triplet explicitly supports PyTorch Euclidean margin loss (including
its epsilon convention) or cosine margin loss, with independent normalization.

Positive candidates are valid distinct in-batch same-clique tracks. Anchors
default to originals for metric terms; choose `all` explicitly for other
policies. Identical sampled track IDs are reduced to their first occurrence
in metric terms; CE/ArcFace retain sampled occurrences. Augmented tracks with
different IDs remain distinct samples. Invalid-positive terms are finite
differentiable zeros, not a global gate on SupCon/ArcFace. Optional historical
skip-batch behavior is an explicit switch. An epoch with no valid optimizer
updates fails instead of exporting a success-shaped run.

Static mining fixes different-clique identities while independently refreshing
embeddings every eight epochs. Dynamic mining reselects every three epochs.
Both initially build at epoch 0. Null intervals mean no later refresh.
Selection is chunked cosine search over **all training tracks**, including
covers, and never over validation/test. Random negatives are independent
different-clique training draws and may equal the similar negative. Cached
negatives are detached; `live` recomputes selected negatives through the model
with gradients. Identity and embedding refresh events are logged separately.

Balanced batches require P distinct cliques and K tracks each. Default 16 x 4.
Original inclusion, error/replacement shortages, duplicate exclusion, tail
drop/error, deterministic seed+epoch ordering and fixed epoch length are
documented explicitly. One original is guaranteed per selected clique, not
every original or every large-clique track in each epoch.

## Rows and initializers

Prerequisites are independent, not a mandatory chain:

| Preset | Aliases / policy | Budget and initialization |
|---|---|---|
| `upstream.ce` | `upstream` | Configurable upstream CE, scratch or explicit warm start |
| `prerequisites.ce` | Domain CE | Reconstructed base budget, scratch or warm start |
| `prerequisites.triplet` | Euclidean default, optional cosine | Scratch or explicit initializer |
| `prerequisites.modified_static` | `Timothee`, `modified_static` | Source-observed Indian: 30 epochs, patience 15, LR 1e-4, embedding refresh 8 |
| `prerequisites.modified_dynamic` | `modified_dynamic` | Source-observed Indian: 150 epochs, patience 25, LR 5e-5, refresh 3 |

Timothee is an alias for static real different-clique hard negatives, not an
augmented original mislabeled negative and not an invented literature attribution.

All Phase 1 rows explicitly initialize from the appropriate domain/modality
dynamic-modified checkpoint. `--init` is required; the dependency is not guessed.

| Preset | Original/thesis | Enabled components |
|---|---|---|
| `phase1.r1` | R1/C1 | Dynamic modified triplet |
| `phase1.r2` | R2/C2 | Tonic + modified |
| `phase1.r3` | R3/C3 | Modified + SupCon |
| `phase1.r4` | R4/C4 | Modified + ArcFace |
| `phase1.r5` | R5/C5 | Tonic + modified + SupCon |
| `phase1.r6` | R6/C6 | Tonic + modified + ArcFace |
| `phase1.r7` | R7/C7 | Tonic + modified + SupCon + ArcFace |
| `phase1.r8` | R8/C8 | Tonic only, zero optimization epochs |
| `phase1.r9` | R9/C9 | Modified + SupCon + ArcFace, no tonic |

Every Phase 2 row independently initializes from R9; all retain the underlying
dynamic modified objective. ArcFace is off unless the row says otherwise.

| Preset | Aliases | Positive / SupCon / batches | Epochs |
|---|---|---|---|
| `phase2.e0` | E0/S0 | Random / 0.5 / random | 10 |
| `phase2.e1` | E1/S1 | Random / 0.5 / balanced | 10 |
| `phase2.e2` | E2/S2 | Mean / off / random | 10 |
| `phase2.e2_bal` | E2_bal/S2b | Mean / off / balanced | 10 |
| `phase2.e3` | E3/S3 | Hard / 0.5 / random | 10 |
| `phase2.e3b` | E3b/S3b | Hard / off / random (historically unverified) | 10 |
| `phase2.e4` | E4/S4 | Hard+mean / off / balanced | 20 |
| `phase2.e5` | E5/S5 | Hard+mean / 0.5 / balanced | 20 |
| `phase2.e6` | E6/S6 | Hard / 0.5 / balanced | 10 |

E4/E5 20-epoch budgets are user-confirmed. Each Phase 3 row independently starts
from R9, **not** a screening checkpoint:

| Preset | Aliases | Configuration |
|---|---|---|
| `phase3.e4_75` | E4_75/F4 | E4 components, 75 epochs |
| `phase3.e3_75` | E3_75/F3 | E3 components, 75 epochs |
| `phase3.e4_arc` | E4_arc/F4_arc | E4_75 + ArcFace |
| `phase3.e3_arc` | E3_arc/F3_arc | E3_75 + ArcFace |
| `phase3.e4_sup_low` | E4_sup_low/F4_sup | E4_75 + SupCon weight 0.15 |

Report settings: Adam; LR 5e-4 in Phase 1, 5e-5 in Phase 2/3; weight decay 1e-4;
projection dropout 0.3; cosine minimum 1e-6/horizon budget; batch 64;
Conv0 frozen with BatchNorm explicitly `train`; refresh 3; patience 15;
minimum improvement 0.002; ArcFace 3 subcenters/margin 0.3/scale 30.
The batchnorm, anchor/cache/duplicate policies, dropout placement, split
seed/rounding, and unresolved long-run stopping overrides are reconstructed.
SupCon temperature 0.07 and ArcFace weight 0.5 are pre-final observed defaults,
not established final-run facts.
The ArcFace sine calculation has an explicit 1e-12 numerical floor to keep
gradients finite at unit cosine; this guard does not modify the backbone.

## Targets and remaining evidence gaps

[report_targets.json](../src/csi/presets/report_targets.json) stores the seven
principal MAP rows supplied in the plan, fixed column order, earlier prerequisite
values, an explicit conflicting SHS-vocal modified result (0.6717 versus
0.6005), and earlier non-final ablation values. Remaining final Phase 1/2 values
are explicitly unrecovered, not invented. Target comparisons are descriptive;
they never change loss/training behavior and do not prove reproduction.

Also unresolved: exact validation membership, initialization checkpoint hashes,
CE exposure to later held-out cliques, small-clique policies in final runs,
final negative cache semantics, augmentation composition, historical runtime,
and the 2,149-track count. Source-observed, report-specified, user-confirmed and
reconstructed choices must not be conflated.

Retrieval normalizes embeddings and ranks cosine similarity with self excluded;
ties preserve manifest order. AP averages relevant-result precision; Top10
always divides by 10; Rank1 is mean first relevant rank, not reciprocal rank.
The observed legacy no-positive convention is zero for AP/Top10/Rank1; use
`evaluation.no_positive=error` to reject such queries. Zero-norm/non-finite
embeddings are explicit errors, not arbitrary cosine rankings.
