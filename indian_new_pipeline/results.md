# Indian New Pipeline — Results Summary

All runs use `cqtnet_indian_best.pth` as the starting checkpoint (pre-trained on SHS100K, fine-tuned on Indian data with CE).

## Test Set Performance

| Method | MAP | Top10 | Rank1 |
|--------|-----|-------|-------|
| CE Baseline (no fine-tuning) | 0.7366 | 0.3811 | 6.42 |
| Triplet Loss | **0.7460** | **0.3825** | **4.40** |
| Dynamic Modified Triplet Loss | 0.7620 | 0.5566 | 4.53 |

## Notes

- **CE Baseline**: checkpoint loaded directly, no stage-2 fine-tuning.
- **Triplet Loss**: standard triplet with random anchor/positive/negative construction (`TripletMarginLoss`, margin=0.05).
- **Dynamic Modified Triplet Loss**: modified loss `L = max(0, margin + 2·d(A,P) − [d(A,N_sim) + d(A,N_rand)])` (margin=0.3) with N_sim recomputed from current model embeddings every 3 epochs. Best checkpoint at ~104 epochs (`CQTNet_indian_dyn_triplet_best_indian_fin_map76.pth`), val MAP=0.7620.
- Static modified triplet loss (N_sim precomputed once from initial checkpoint) was tested but showed consistent val_MAP degradation from epoch 2 onwards and was abandoned.
