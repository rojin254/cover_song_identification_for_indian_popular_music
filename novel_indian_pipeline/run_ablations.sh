#!/bin/bash
# 8-row ablation matrix for the novel Indian CSI pipeline.
#
# Each row trains TWO models — one on Indian-mix, one on Indian-vocal — then
# evaluates each on all four test manifests. Output goes to
# novel_indian_pipeline/results/tables/<row_name>__trainedon_<dataset>/.
#
# Re-run safe: each invocation writes a unique timestamped log + ckpt; the
# results CSVs in results/tables/<row_name>... are overwritten on re-run.
#
# Prereqs:
#   1. Copy .env.example to .env.
#   2. Run tonic_index_build for every dataset that any tonic-rotated row
#      will touch (rows 2, 5, 6, 7, 8 — all four datasets, since eval also
#      uses the rotated CQT for those rows).
#
# Usage:
#   bash novel_indian_pipeline/run_ablations.sh           # run all 8 rows on both datasets
#   bash novel_indian_pipeline/run_ablations.sh row1      # run row 1 only
#   bash novel_indian_pipeline/run_ablations.sh row7 indian_vocal   # run row 7 on vocal only
#
set -euo pipefail

cd "$(dirname "$0")/.."        # PP4/

PY=${PY:-python}
PIPE=novel_indian_pipeline

ROW_FILTER=${1:-all}
DATASET_FILTER=${2:-both}

run() {
    # run <row_name> <dataset> <flag args>
    local row="$1"; shift
    local dataset="$1"; shift
    if [[ "$ROW_FILTER" != "all" && "$ROW_FILTER" != "$row" ]]; then
        return 0
    fi
    if [[ "$DATASET_FILTER" != "both" && "$DATASET_FILTER" != "$dataset" ]]; then
        return 0
    fi
    echo ""
    echo "============================================================"
    echo "  ROW $row   |   dataset=$dataset"
    echo "  args: $*"
    echo "============================================================"
    "$PY" -m $PIPE.main train \
        --dataset="$dataset" \
        --row_name="$row" \
        "$@"
}

ensure_tonic_index() {
    local ds="$1"
    local idx="$PIPE/tonic_index/${ds}.json"
    if [[ ! -f "$idx" ]]; then
        echo "Building tonic index for $ds ..."
        "$PY" -m $PIPE.tonic_index_build --dataset="$ds"
    fi
}

# Tonic indices needed for any tonic_normalize=True row.
ensure_tonic_index indian
ensure_tonic_index indian_vocal
ensure_tonic_index shs100k
ensure_tonic_index shs100k_vocal

for DS in indian indian_vocal; do
    # Row 1: baseline reproduction
    run row1_baseline_dyn_triplet "$DS"
    # Row 2: tonic + dyn-triplet
    run row2_tonic_dyn_triplet "$DS" --tonic_normalize=True
    # Row 3: dyn-triplet + supcon
    run row3_dyn_triplet_supcon "$DS" --use_supcon=True
    # Row 4: dyn-triplet + arcface
    run row4_dyn_triplet_arcface "$DS" --use_arcface=True
    # Row 5: tonic + dyn-triplet + supcon
    run row5_tonic_dyn_triplet_supcon "$DS" --tonic_normalize=True --use_supcon=True
    # Row 6: tonic + dyn-triplet + arcface
    run row6_tonic_dyn_triplet_arcface "$DS" --tonic_normalize=True --use_arcface=True
    # Row 7: full recipe (HEADLINE)
    run row7_full "$DS" --tonic_normalize=True --use_supcon=True --use_arcface=True
    # Row 8: tonic only, no dyn-triplet (sanity)
    run row8_tonic_only "$DS" --tonic_normalize=True --use_dyn_triplet=False
done

echo ""
echo "All requested rows finished. Building summary..."
"$PY" -m $PIPE.main build_summary
