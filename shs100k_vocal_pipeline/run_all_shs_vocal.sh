#!/bin/bash
set -e

echo "Starting SHS100k vocal pipeline (CE + Triplet + Dynamic Modified Triplet)..."
cd /teamspace/studios/this_studio/PP4/shs100k_vocal_pipeline
/home/zeus/miniconda3/envs/cloudspace/bin/python main.py transfer_learning
echo "SHS100k vocal pipeline completed successfully."