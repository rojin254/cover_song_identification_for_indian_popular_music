#!/bin/bash
# Runs the missing indian_vocal training for rows 3-9.
# Rows 1 and 2 already have indian_vocal results from the original script run.
# Run this after row9 on indian finishes.
#
# Usage:
#   bash novel_indian_pipeline/run_vocal_missing.sh
#   bash novel_indian_pipeline/run_vocal_missing.sh row5   # single row
set -euo pipefail

cd "$(dirname "$0")/.."   # PP4/

PY=${PY:-python}
PIPE=novel_indian_pipeline
ROW_FILTER=${1:-all}
DS=indian_vocal

run() {
    local row="$1"; shift
    if [[ "$ROW_FILTER" != "all" && "$ROW_FILTER" != "$row" ]]; then
        return 0
    fi
    echo ""
    echo "============================================================"
    echo "  ROW $row   |   dataset=$DS"
    echo "  args: $*"
    echo "============================================================"
    "$PY" -m $PIPE.main train \
        --dataset="$DS" \
        --row_name="$row" \
        "$@"
}

# Row 3: dyn-triplet + supcon
run row3_dyn_triplet_supcon --use_supcon=True
# Row 7: full recipe
run row7_full --tonic_normalize=True --use_supcon=True --use_arcface=True
# Row 9: dyn-triplet + supcon + arcface (no tonic)
run row9_dyn_triplet_supcon_arcface --use_supcon=True --use_arcface=True
# Row 4: dyn-triplet + arcface
run row4_dyn_triplet_arcface --use_arcface=True
# Row 5: tonic + dyn-triplet + supcon
run row5_tonic_dyn_triplet_supcon --tonic_normalize=True --use_supcon=True
# Row 6: tonic + dyn-triplet + arcface
run row6_tonic_dyn_triplet_arcface --tonic_normalize=True --use_arcface=True
# Row 8: tonic only, no dyn-triplet (sanity)
run row8_tonic_only --tonic_normalize=True --use_dyn_triplet=False

echo ""
echo "All vocal rows finished. Building summary..."
"$PY" -m $PIPE.main build_summary
