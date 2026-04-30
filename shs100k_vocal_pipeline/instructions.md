# SHS100k Vocal Pipeline

This pipeline is dedicated to vocal-separated SHS100k data and performs triplet-loss finetuning only.

## Data Inputs
- Train audio: `data/train/shs100k_train_vocal/*.mp3`
- Test audio: `data/test/shs100k_test_vocal/*.mp3`
- Metadata: `data/shs100k-train.csv`, `data/shs100k-test.csv`

## Steps
1. Build train/test txt files (`filename,clique,is_original`):
```bash
cd shs100k_vocal_pipeline
python prepare_data.py
```

2. Generate CQT features:
```bash
python generate_cqt.py
```

3. Run triplet finetuning from the best base CQTNet checkpoint:
```bash
python main.py run_triplet_finetune
```

## Notes
- Default initialization checkpoint: `check_points/cqtnet_shs100k_best.pth`
- You can continue finetuning from the earlier triplet model:
```bash
python main.py run_triplet_finetune --load_model_path="/teamspace/studios/this_studio/PP4/check_points/CQTNet_SHS100k_finetune_epoch_74.pth"
```
- CQT arrays are saved under `shs100k_vocal_pipeline/cqt_features/`.
