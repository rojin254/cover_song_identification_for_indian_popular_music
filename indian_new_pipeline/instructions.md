# Data Preparation Instructions

This document explains how to recreate the dataset preparation pipeline for the new Indian cover song dataset.

## Directory Structure
Ensure you have the raw audio files and metadata CSV files in the following structure:
- `data/train/indian/*.mp3`
- `data/test/indian/*.mp3`
- `data/indian_train.csv`
- `data/indian_test.csv`

All the new scripts and output data generated for this pipeline are stored in the `indian_new_pipeline` folder to isolate them from the existing pipeline.

## Execution Steps

1. **Install Requirements**
   Make sure you have `librosa`, `pandas`, and `tqdm` installed in your Python environment:
   ```bash
   pip install librosa pandas tqdm
   ```

2. **Run Data Preparation**
   Execute the `prepare_data.py` script. This script will read the CSV files, extract the Constant-Q Transform (CQT) features from the audio files, and generate the data splits.
   ```bash
   cd indian_new_pipeline
   python prepare_data.py
   ```

   **What the script does:**
   - Parses `data/indian_train.csv` and `data/indian_test.csv` to generate `indian_train.txt` and `indian_test.txt` which contain the list of IDs (e.g. `100-1-ci-c`) without file extensions.
   - Iterates through all `.mp3` files in the source directories.
   - Uses `librosa.load` with a target sampling rate of 22050 Hz and `librosa.cqt` to extract the CQT arrays.
   - Saves each CQT matrix as a NumPy (`.npy`) array to the `indian_new_pipeline/cqt_features/` directory.

3. **Codebase Updates (Already Handled)**
   The core training scripts have been updated to utilize this new dataset pipeline:
   - **`cqt_loader.py`**: Reads from the newly generated `indian_train.txt`/`indian_test.txt` text splits, targets the `cqt_features/` folder, and extracts a boolean flag `is_original` indicating if the feature is an original song or a cover.
   - **`main.py`**: The `custom_collate` and data loading loops have been updated to propagate the `is_original` flag. The triplet loss function `create_triplets` now utilizes this boolean to guarantee that anchors are always original tracks and handles an arbitrary number of covers for each original anchor. Finally, the model classification head dimension has been changed to `402` to match the number of distinct cliques found in the dataset.

After completing the data preparation, you can execute the training process by running `python main.py transfer_learning --fine_tune=True`.
