#!/bin/bash
# Run vocal separation for all datasets
# Usage: bash scripts/vocalseparation/run_vocal_sep.sh

set -e

cd "$(dirname "$0")/../.."

echo "============================================"
echo "  Vocal Separation Pipeline (Demucs htdemucs)"
echo "============================================"

# Install demucs if not already installed
pip install -q demucs 2>/dev/null || pip install demucs

# Run separation for each dataset
echo ""
echo ">>> Step 1/3: Indian Test Set"
python scripts/vocalseparation/separate_vocals.py --dataset indian_test --workers 4

echo ""
echo ">>> Step 2/3: SHS100K Train Set"
python scripts/vocalseparation/separate_vocals.py --dataset shs100k_train --workers 4

echo ""
echo ">>> Step 3/3: SHS100K Test Set"
python scripts/vocalseparation/separate_vocals.py --dataset shs100k_test --workers 4

echo ""
echo "============================================"
echo "  All vocal separations complete!"
echo "============================================"
echo ""
echo "Output directories:"
echo "  data/test/indian_vocal/"
echo "  data/train/shs100k_train_vocal/"
echo "  data/test/shs100k_test_vocal/"
