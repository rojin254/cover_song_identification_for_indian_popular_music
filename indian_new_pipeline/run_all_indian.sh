#!/bin/bash
set -e

echo "Starting Indian data preparation..."
cd /teamspace/studios/this_studio/PP4/indian_new_pipeline
python prepare_data.py

echo "Starting Indian training (CE + Triplet)..."
cd /teamspace/studios/this_studio/PP4/indian_new_pipeline
# python main.py transfer_learning --fine_tune=True

# echo "Indian pipeline completed successfully."

# echo "Starting Indian pipeline with modified triplet loss..."
# cd /teamspace/studios/this_studio/PP4/indian_new_pipeline
# python main.py transfer_learning --modified_triplet=True --skip_transfer=True

echo "Starting Indian pipeline with dynamic triplet loss..."
cd /teamspace/studios/this_studio/PP4/indian_new_pipeline
python main.py transfer_learning --dynamic_triplet=True --skip_transfer=True
