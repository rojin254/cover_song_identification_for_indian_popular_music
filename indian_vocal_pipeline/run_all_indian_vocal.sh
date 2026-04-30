#!/bin/bash
set -e

echo "Starting Indian vocal data preparation..."
cd /teamspace/studios/this_studio/PP4/indian_vocal_pipeline
python prepare_data.py

echo "Starting Indian vocal training (CE + Triplet)..."
cd /teamspace/studios/this_studio/PP4/indian_vocal_pipeline
python main.py transfer_learning --fine_tune=True

echo "Indian vocal pipeline completed successfully."

# echo "Starting Indian vocal pipeline with modified triplet loss..."
# cd /teamspace/studios/this_studio/PP4/indian_vocal_pipeline
# python main.py transfer_learning --modified_triplet=True --skip_transfer=True