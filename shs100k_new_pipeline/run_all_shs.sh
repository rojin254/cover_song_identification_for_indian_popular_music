#!/bin/bash
set -e

echo "Starting Data Preparation..."
cd /teamspace/studios/this_studio/PP4/shs100k_new_pipeline
/home/zeus/miniconda3/envs/cloudspace/bin/python prepare_data.py
echo "Data Preparation Complete."

echo "Starting CQT Generation..."
cd /teamspace/studios/this_studio/PP4/shs100k_new_pipeline
/home/zeus/miniconda3/envs/cloudspace/bin/python generate_cqt.py
echo "CQT Generation Complete."

echo "Starting Model Fine-Tuning and Testing..."
cd /teamspace/studios/this_studio/PP4/shs100k_new_pipeline
/home/zeus/miniconda3/envs/cloudspace/bin/python main.py transfer_learning --skip_transfer=True --fine_tune=True
echo "All Processes Completed Successfully."


# echo "Starting SHS100K pipeline with modified triplet loss..."
# cd /teamspace/studios/this_studio/PP4/shs100k_new_pipeline
# python main.py transfer_learning --modified_triplet=True --skip_transfer=True