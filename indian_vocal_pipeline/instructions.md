# Indian Vocal Pipeline

This pipeline is dedicated to vocal-separated Indian cover data and performs triplet-loss finetuning only.

## Data Inputs
- Train audio: `data/train/indian_vocal_train/*.mp3`
- Test audio: `data/test/indian_vocal/*.mp3`
- Metadata: `data/indian_train.csv`, `data/indian_test.csv`

## Steps
1. Generate CQT features and split files:
```bash
cd indian_vocal_pipeline
python prepare_data.py
```

2. Run triplet finetuning from base checkpoint:
```bash
python main.py run_triplet_finetune
```

## Notes
- Default initialization checkpoint: `check_points/cqtnet_indian_best.pth`
- You can override it:
```bash
python main.py run_triplet_finetune --load_model_path="/teamspace/studios/this_studio/PP4/check_points/CQTNet_transfer_learning_epoch_64.pth"
```
- CQT arrays are saved under `indian_vocal_pipeline/cqt_features/`.
