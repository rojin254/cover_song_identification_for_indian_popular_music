#!/bin/bash
set -e

echo "Starting Data Preparation..."
cd /teamspace/studios/this_studio/PP4/indian_new_pipeline
/home/zeus/miniconda3/envs/cloudspace/bin/python prepare_data.py
echo "Data Preparation Complete."

echo "Starting Model Fine-Tuning..."
cd /teamspace/studios/this_studio/PP4
/home/zeus/miniconda3/envs/cloudspace/bin/python main.py transfer_learning --fine_tune=True --skip_transfer=True
echo "All Processes Completed Successfully."
