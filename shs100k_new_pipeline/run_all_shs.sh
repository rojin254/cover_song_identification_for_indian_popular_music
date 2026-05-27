#!/bin/bash
set -e

echo "Starting SHS100K pipeline with modified triplet loss..."
cd /teamspace/studios/this_studio/PP4/shs100k_new_pipeline
/home/zeus/miniconda3/envs/cloudspace/bin/python main.py transfer_learning --modified_triplet=True --skip_transfer=True
echo "All Processes Completed Successfully."