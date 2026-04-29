# SHS100K Dataset Run Instructions

This pipeline is adapted specifically to handle the `shs100k` dataset, maintaining all previous progress for the Indian Cover Songs dataset, but providing isolated components to reproduce the results.

## Prerequisites

Ensure that the `.mp3` datasets are present in the following directories as documented in the dataset configuration:
- Training audio: `data/train/shs100k_train/`
- Testing audio: `data/test/shs100k-test/`
- Metadata: `data/shs100k-train.csv` and `data/shs100k-test.csv`

## Execution Steps

### 1. Prepare Text Lists
Run the `prepare_data.py` script. This script reads the provided CSVs and outputs `.txt` files containing `filename, clique, is_original` formatted for the dataset loaders. The original anchor song per clique is marked using the lowest `version` present for each clique.

```bash
python shs100k_new_pipeline/prepare_data.py
```

### 2. Generate CQT Features
Run the feature extraction script. This uses multiprocessing to read the MP3 files from the raw data directories and outputs `.npy` Constant-Q Transform matrices. They are saved in `shs100k_new_pipeline/cqt_features/`.

```bash
python shs100k_new_pipeline/generate_cqt.py
```

### 3. Run Training and Evaluation
Run the modified `main.py` tailored for the `SHS100K` set. It imports `SHS100kCQT` from `cqt_loader.py` and maps targets appropriately to the updated classifier layer dimensions. It also adapts to cliques of variable lengths seamlessly.

```bash
python shs100k_new_pipeline/main.py transfer_learning
```

If you wish to fine-tune using Triplet Loss subsequently, provide the fine-tune flag:
```bash
python shs100k_new_pipeline/main.py transfer_learning --fine_tune=True
```

## Structure
- `prepare_data.py`: Prepares metadata.
- `generate_cqt.py`: Pre-computes `.npy` from raw audio.
- `cqt_loader.py`: A `torch.utils.data.Dataset` mapping labels and retrieving the arrays.
- `main.py`: Contains training, triplet creation logic (independent of fixed clique length), and evaluations.
