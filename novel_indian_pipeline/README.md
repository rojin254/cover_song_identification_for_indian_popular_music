# novel_indian_pipeline

A self-contained pipeline that adds two ideas on top of the existing Indian
CSI baseline:

1. **Tonic-relative CQT rotation** as preprocessing — tested across 5 rows,
   consistently hurt MAP by 5–8%; treated as a negative result (see
   [research.md](research.md)).
2. **SupCon + SubCenter-ArcFace + dynamic-modified-triplet** as the loss —
   SupCon is the dominant improvement (+0.75% MAP on Indian, +1.57% on SHS100K
   over the baseline).

The pipeline ran **9 ablation rows** in total. Best result so far: **Row 3
(Dyn Triplet + SupCon)** at MAP 0.7500 Indian / 0.8466 SHS100K. Row 9 (SupCon
+ ArcFace, no tonic) is the latest experiment.

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

# 3. Run the full 9-row ablation matrix on both Indian datasets.
bash novel_indian_pipeline/run_ablations.sh

# Or run a single row:
bash novel_indian_pipeline/run_ablations.sh row9 indian

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

## Ablation rows

| Row | Config | Indian MAP | SHS100K MAP |
|-----|--------|:----------:|:-----------:|
| 1 | Dyn Triplet (baseline) | 0.7425 | 0.8309 |
| 2 | Tonic + Dyn Triplet | 0.6597 | 0.7624 |
| **3** | **Dyn Triplet + SupCon** | **0.7500** | **0.8466** |
| 4 | Dyn Triplet + ArcFace | 0.7367 | 0.8290 |
| 5 | Tonic + Dyn Triplet + SupCon | 0.6980 | 0.7945 |
| 6 | Tonic + Dyn Triplet + ArcFace | 0.6515 | 0.7745 |
| 7 | Tonic + Dyn Triplet + SupCon + ArcFace | 0.6774 | 0.7862 |
| 8 | Tonic only (sanity) | 0.6061 | 0.7350 |
| 9 | Dyn Triplet + SupCon + ArcFace | *running* | *running* |

All rows trained on Indian-mix, 75 epochs max, patience 15.
