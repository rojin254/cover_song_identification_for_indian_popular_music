#!/bin/bash
set -e
cd /teamspace/studios/this_studio/PP4
/home/zeus/miniconda3/envs/cloudspace/bin/python shs100k_vocal_pipeline/main.py transfer_learning \
  --skip_transfer=True \
  --load_model_path=/teamspace/studios/this_studio/PP4/check_points/CQTNet_shs100k_vocal_ce_best_shs100k_vocal_20260502_191055.pth \
  2>&1 | tee /teamspace/studios/this_studio/PP4/shs100k_vocal_pipeline/logs/run_from_ce_ckpt.log
