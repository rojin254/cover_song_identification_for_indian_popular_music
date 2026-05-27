# novel_indian_pipeline

A self-contained pipeline that adds two ideas on top of the existing Indian
CSI baseline:

1. **Tonic-relative CQT rotation** as preprocessing — see
   [research.md](research.md) and
   [NOVEL_INDIAN_CSI_PROPOSAL.md](NOVEL_INDIAN_CSI_PROPOSAL.md).
2. **SupCon + SubCenter-ArcFace + dynamic-modified-triplet** as the loss.

Nothing in `indian_new_pipeline/`, `indian_vocal_pipeline/`,
`shs100k_new_pipeline/`, or `shs100k_vocal_pipeline/` is modified. CQT
features and warm-start checkpoints are referenced — not copied — through
paths in `.env`.

## Quickstart

```bash
cd /teamspace/studios/this_studio/PP4

# 1. Configure paths (edit if your layout differs).
cp novel_indian_pipeline/.env.example novel_indian_pipeline/.env
$EDITOR novel_indian_pipeline/.env

# 2. Precompute tonic indices (one-time, ~ms per file).
python -m novel_indian_pipeline.tonic_index_build --dataset=indian
python -m novel_indian_pipeline.tonic_index_build --dataset=indian_vocal
python -m novel_indian_pipeline.tonic_index_build --dataset=shs100k
python -m novel_indian_pipeline.tonic_index_build --dataset=shs100k_vocal

# 3. Run the full 8-row ablation matrix on both Indian datasets.
bash novel_indian_pipeline/run_ablations.sh

# Or run a single row:
bash novel_indian_pipeline/run_ablations.sh row7 indian_vocal

# 4. After all rows finish, summarise.
python -m novel_indian_pipeline.main build_summary
```

## Layout

```
novel_indian_pipeline/
  .env / .env.example          # paths to external CQT dirs, manifests, ckpts
  env_paths.py                 # tiny .env loader + dataset registry
  NOVEL_INDIAN_CSI_PROPOSAL.md # formal research proposal
  research.md                  # plain-language version with paper links
  tonic.py                     # estimate_tonic_from_cqt(cqt) -> int
  cqt_rotate.py                # rotate_cqt(cqt, tonic_offset)
  tonic_index_build.py         # one-time per-dataset tonic JSON builder
  cqt_loader.py                # GenericCQT serving all four datasets
  models/CQTNet.py             # verbatim CQTNet (300-d embedding)
  losses.py                    # modified_triplet + supcon + SubCenterArcFace
  main.py                      # train / evaluate / build_summary commands
  run_ablations.sh             # drives the 8-row matrix
  tonic_index/                 # JSON sidecars from tonic_index_build.py
  check_points/                # newly-trained checkpoints (this pipeline only)
  logs/                        # per-run JSON metric logs
  results/
    tables/<row>__trainedon_<dataset>/
      indian_test.csv          # MAP / Top10 / MR1 / n_queries
      indian_vocal_test.csv
      shs100k_test.csv
      shs100k_vocal_test.csv
      all_datasets_summary.csv
    tables/summary.csv         # auto-aggregated by `main.py build_summary`
    ablation_matrix.md         # human-readable summary
```

## Datasets

Per the project's restriction, only these manifests are used:

| Dataset key       | Use                                  |
|-------------------|--------------------------------------|
| `indian`          | training + evaluation (mix)          |
| `indian_vocal`    | training + evaluation (vocal stems)  |
| `shs100k`         | evaluation only (mix, train+test)    |
| `shs100k_vocal`   | evaluation only (vocals, train+test) |

Covers80 is **not** used.

## Commands

```bash
# Train one ablation row (all flags optional; see main.py docstring).
python -m novel_indian_pipeline.main train \
    --dataset=indian_vocal \
    --row_name=row7_full \
    --tonic_normalize=True \
    --use_supcon=True \
    --use_arcface=True

# Eval-only on a saved checkpoint.
python -m novel_indian_pipeline.main evaluate \
    --dataset=indian_vocal \
    --checkpoint=novel_indian_pipeline/check_points/CQTNet_row7_full_indian_vocal_..._best.pth \
    --tonic_normalize=True

# Aggregate all per-row results into one summary.
python -m novel_indian_pipeline.main build_summary
```

## Reproduction target

Row 1 (`row1_baseline_dyn_triplet` on `indian_vocal`) must reproduce the
existing best result from [ISMIR_CSI_PAPER_CONTENT.md](
../ISMIR_CSI_PAPER_CONTENT.md) — MAP ≈ 0.9234 on Indian-vocal — within ±0.005.
This validates the refactor before any of the new components are switched on.
