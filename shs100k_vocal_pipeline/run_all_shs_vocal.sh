#!/bin/bash
set -e

echo "Preparing SHS100k vocal split files..."
cd /teamspace/studios/this_studio/PP4/shs100k_vocal_pipeline
python prepare_data.py

echo "Generating SHS100k vocal CQT features..."
cd /teamspace/studios/this_studio/PP4/shs100k_vocal_pipeline
python generate_cqt.py

echo "Starting SHS100k vocal training (CE + Triplet)..."
cd /teamspace/studios/this_studio/PP4/shs100k_vocal_pipeline
python main.py transfer_learning --fine_tune=True

echo "SHS100k vocal pipeline completed successfully."

# echo "Starting SHS100k vocal pipeline with modified triplet loss..."
# cd /teamspace/studios/this_studio/PP4/shs100k_vocal_pipeline
# python main.py transfer_learning --modified_triplet=True --skip_transfer=True