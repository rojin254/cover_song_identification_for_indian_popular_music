# Remote execution and verification

## Status and safety

Verified on 2026-10-25 on the authorized Linux host with Python 3.12.3 and
PyTorch 2.14.1+cpu: **40 tests passed after cleanup**, with three upstream audioread
deprecation warnings. Whitespace checks passed. Wheel and source distribution
builds passed; the wheel's packaged presets loaded after installation outside
the repository, and the source distribution includes documentation and the
pinned reference fixture without legacy pipelines. Python 3.11/3.12 CI is
configured, but hosted CI itself has not been run in this session. CUDA is
unavailable in the installed CPU runtime.

The pre-cleanup baseline passed 39 tests. A new installed-command smoke test
then ran `csi` and `python -m csi` outside the repository through synthetic
preparation, preflight, CPU training, completed-budget resume, checkpoint
evaluation, four-set reevaluation/integrity verification and summary. All
temporary fixture/run artifacts are cleaned up by the test. The resume command
in that CLI smoke deliberately adds no epochs; interrupted-versus-uninterrupted
training is tested separately below.

CPU tests cover the installed parser/schema, every preset/alias, independent
objectives, exact backbone parity, modern audio extraction, fixture preparation,
preflight, CE/ordinary/static/dynamic training, upstream three-length CE,
four-way evaluation, artifact verification and exact epoch resume.
Tests also resume dropout/balanced/SupCon/ArcFace through the epoch-3 negative
refresh boundary and compare model, heads, optimizer, scheduler, RNG, sampler,
negative cache, histories and patience.

Full dataset recovery, final checkpoint recovery, GPU smoke tests, historical
CQT audio parity and report reruns are **not performed**. No paid compute,
full-data download, full training suite, upload, commit or push was started.
At the user's explicit request, legacy cleanup was moved ahead of GPU/full-data
checks: 250 unchanged tracked legacy files (637.9 MiB) were removed from this
branch after the CPU baseline passed. The original branch, historical Git
objects and user's rebuild plan are preserved. Cleanup is not a claim of
historical report reproduction.

## Install

On the explicitly authorized remote host:

```bash
uv sync --frozen --extra audio --extra test
uv run csi --help
uv run python -m csi list
uv run --extra audio --extra test pytest tests -q
```

The lock defaults to CPU PyTorch and records the modern runtime. To use an
already authorized GPU host, create a separate GPU environment rather than
silently replacing the CPU lock:

```bash
python3 -m venv .venv-gpu
.venv-gpu/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv-gpu/bin/pip install -e '.[audio,test]'
.venv-gpu/bin/csi doctor --preset prerequisites.ce --set train.device=cuda
```

Choose a supported wheel/CUDA pair for the actual host; the above GPU example
is not verified. Do not use default `uv run` to resync the GPU environment to
CPU. Set `train.device=cuda` explicitly. No distributed or mixed-precision
historical reproduction is claimed.

## First complete synthetic run

```bash
uv run csi data fixture --output data/fixture
uv run csi data prepare --inventory data/fixture/metadata/tracks.csv \
  --set paths.data=data/fixture
uv run csi doctor --preset prerequisites.ce \
  --set paths.data=data/fixture --set train.batch_size=4
uv run csi train --preset prerequisites.ce --run-id fixture-ce \
  --set paths.data=data/fixture --set preprocess.frames=200 \
  --set train.batch_size=4 --set train.epochs=2
uv run csi evaluate --preset prerequisites.ce --run-id fixture-eval \
  --set paths.data=data/fixture --set preprocess.frames=200 \
  --init runs/fixture-ce/checkpoints/best.pt
uv run csi verify --run runs/fixture-ce --reevaluate \
  --output-run-id fixture-parity --atol 1e-8
uv run csi summarize --runs runs --output fixture-summary.json
```

To rerun just the CPU lifecycle smoke tests, including the installed CLI running
outside the repository:

```bash
uv run --extra audio --extra test pytest tests/test_lifecycle.py tests/test_workflows.py -q
```

Use a fresh run ID each time. A fixture contains 32 synthetic tracks across
both datasets/modalities with explicit disjoint train/val/test cliques. The
fixture's retrieval MAP is not a report result.

## Recover assets without a full download

Select exact verified remote paths; examples below are metadata, not audio:

```bash
uv run csi data fetch --asset dataset --file coversIndian_list.txt \
  --output data/metadata/recovered
```

For audio or checkpoint files, use explicit known paths and the correct
destination from [DATA](DATA.md). Do not presume final regularized checkpoints
exist in the earlier checkpoint repository. Translate preserved historical
membership into the CSV inventory, verify clique/family isolation and
cross-modality identity, then:

```bash
uv run csi data prepare --inventory data/metadata/tracks.csv --features \
  --set train.workers=4
uv run csi doctor --preset prerequisites.ce
```

If tonic rows are selected, build the index once for that training modality
and all four test sets and set `paths.tonic` on the run. Missing test sets,
audio, features, sidecars, hashes, label maps or caches must be fixed, not skipped.

## Prerequisites: independent execution

The paths below are placeholders for **verified explicit** initializer
artifacts. Do not run these training commands until inputs and compute are
authorized. Optional warm starts do not imply a mandatory stage chain.

```bash
uv run csi train --preset upstream.ce --run-id upstream-ce \
  --dataset shs100k --modality mix
uv run csi train --preset prerequisites.ce --run-id indian-ce \
  --init /verified/upstream.pt --set model.replace_classifier=true
uv run csi train --preset prerequisites.triplet --run-id indian-triplet \
  --init /verified/ce.pt --set objectives.triplet.distance=euclidean
uv run csi train --preset prerequisites.modified_static --run-id indian-static \
  --init /verified/ce.pt --set negative.embedding_refresh_epochs=8
uv run csi train --preset prerequisites.modified_dynamic --run-id indian-dynamic \
  --init /verified/ce.pt --set negative.selection_refresh_epochs=3
```

## Every report row

First inspect `show` and `suite --plan`; inherited settings and uncertainties
are included in each resolved result. Phase 1 runs use the correct explicitly
recorded dynamic checkpoint for Indian mix or vocal. For tonic presets also
provide `--set paths.tonic=...`.

```bash
# Inspect all Phase 1 rows without training.
uv run csi suite --group phase1 --plan

# Execute each row only after authorizing the individual runs.
for row in r1 r2 r3 r4 r5 r6 r7 r8 r9; do
  uv run csi train --preset "phase1.$row" --run-id "indian-mix-$row" \
    --dataset indian --modality mix --init /verified/indian-mix-dynamic.pt \
    --set paths.tonic=data/cache/tonic/indian-mix.json
done

# Repeat independently with Indian-vocal data and its dynamic initializer.
for row in r1 r2 r3 r4 r5 r6 r7 r8 r9; do
  uv run csi train --preset "phase1.$row" --run-id "indian-vocal-$row" \
    --dataset indian --modality vocal --init /verified/indian-vocal-dynamic.pt \
    --set paths.tonic=data/cache/tonic/indian-vocal.json
done

# Every screening row independently uses the SAME Phase 1 R9 initializer.
for row in e0 e1 e2 e2_bal e3 e3b e4 e5 e6; do
  uv run csi train --preset "phase2.$row" --run-id "screen-$row" \
    --init /verified/phase1-r9.pt
done

# Final rows also independently use R9, never a screening checkpoint.
for row in e4_75 e3_75 e4_arc e3_arc e4_sup_low; do
  uv run csi train --preset "phase3.$row" --run-id "final-$row" \
    --init /verified/phase1-r9.pt
done
```

These loops are explicitly expensive instructions, **not executed by this
implementation session**. Equivalent individually selected runs can be
submitted through `suite --execute --init-map ... --run-prefix ...`; no
automatic prerequisite execution is implemented.

## Custom composition, checkpoint state and outputs

```bash
uv run csi train --preset prerequisites.modified_dynamic --run-id custom \
  --init /verified/compatible.pt \
  --set objectives.modified_triplet.positive=hard_mean \
  --set sampler.kind=balanced --set sampler.cliques_per_batch=16 \
  --set sampler.tracks_per_clique=4 --set objectives.supcon.weight=0.15 \
  --set objectives.arcface.weight=0.5 --set preprocess.tonic.enabled=true \
  --set paths.tonic=data/cache/tonic/indian-mix.json
```

SupCon-only or ArcFace-only: set modified-triplet weight to zero and enable the
desired term. Unknown/bad fields fail before training.

Each run exports resolved config/evidence/differences, environment/source
hashes/hardware, initialization lineage, label map, manifest hashes, histories,
raw/weighted losses, valid counts, skip/refresh events, best epoch, metrics,
per-query rankings and an artifact hash ledger.

`last.pt` includes optimizer/scheduler, Python/NumPy/Torch/CUDA RNG,
sampler/epoch, cached embeddings/identities/refresh epochs, current/selected
backbone and heads, history and stopping state. `best.pt` is an explicitly
evaluation-only selected model/head snapshot. Use a fresh run ID and **identical
resolved settings** when resuming:

```bash
uv run csi train --preset prerequisites.ce --run-id fixture-resumed \
  --set paths.data=data/fixture --set preprocess.frames=200 \
  --set train.batch_size=4 --set train.epochs=2 \
  --resume runs/fixture-ce/checkpoints/last.pt
```

A completed-budget resume adds no epochs. To extend the budget, use a warm
start and record a new experiment; changing a cosine horizon is not true resume.
Best selection uses any strict validation MAP increase; patience resets only
after the configured minimum improvement. No test metric selects checkpoints.

## Third-party checklist and troubleshooting

1. Confirm host authorization, branch, source hashes and package environment.
2. Run CPU tests and the synthetic lifecycle before obtaining large assets.
3. Pin recovery revisions; verify exact original membership and every required
   dataset/modality. Check all hashes and unresolved count differences.
4. Verify actual initializers, class mapping, preprocessing and prior clique
   exposure. Raw pretrained states are accepted strictly; evaluation records
   when preprocessing compatibility is unknown.
5. Run GPU smoke checks for CE, ordinary/static/dynamic metric stages, epoch-3
   and epoch-8 refresh, hard+mean balanced low-SupCon with/without ArcFace,
   checkpoint reload and four-way evaluation. This gate is still pending.
6. Reevaluate identical checkpoints first; only then compare reconstructed
   training outputs to reference targets. Do not tune on held-out tests.
7. Inspect `verify`, histories and summary deltas, not just one MAP number.
8. Legacy root/pipeline trainers, reports, logs, binaries, figures and launchers
   have been removed from this branch by explicit user authorization, ahead of
   GPU/full-data gates. Keep the historical references and user's plan intact.
   Rerun CPU tests and build/install checks after any further cleanup; future
   GPU and full-data checks must use the same cleaned toolkit.

If no updates occur, inspect originals, positive membership and balanced
shortages; auxiliary-only configurations are valid without triplets.
If a warm start fails, inspect keys/shapes and explicitly request classifier
replacement only when intended. Never filter unexplained tensors.
If resume fails, restore the exact code/config/data or use a labeled warm start.
If verification fails, retain evidence and fix the changed input/artifact;
do not simply regenerate its seal to hide the failure.
