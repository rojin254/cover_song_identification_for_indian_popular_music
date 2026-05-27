# Indian Vocal Pipeline — Results Summary

All runs use vocal-separated audio as input. The starting checkpoint for fine-tuning stages is the best CE or triplet checkpoint identified below.

## Test Set Performance

| Method | Checkpoint | Test MAP | Test Top10 | Test Rank1 |
|--------|-----------|----------|------------|------------|
| CE Baseline | `ce_best_indian_vocal_20260430_185647` | 0.6811 | 0.3457 | 10.72 |
| Triplet Loss | `triplet_best_indian_vocal_triplet_20260430_183443` | **0.7279** | 0.3628 | 9.68 |
| Dynamic Modified Triplet Loss (ce201641) | `dyn_triplet_best_vocal_dyn_triplet_ce201641_full` | 0.6785 | 0.3440 | 10.23 |

## Checkpoint Comparison (all 4 evaluated)

| Checkpoint | Val MAP | Test MAP | Test Top10 | Test Rank1 |
|-----------|---------|----------|------------|------------|
| `ce_best_185647` | 0.5345 | 0.6811 | 0.3457 | 10.72 |
| `ce_best_201641` | 0.5392 | 0.6802 | 0.3437 | 10.49 |
| `triplet_best_183443` ← **best** | 0.5016 | **0.7279** | 0.3628 | 9.68 |
| `triplet_best_201641` | 0.5416 | 0.6794 | 0.3420 | 10.65 |

## Notes

- **CE Baseline**: `_185647` selected as best CE checkpoint (highest test MAP among CE runs).
- **Triplet Loss**: `_183443` selected as best overall checkpoint; triplet run 2 (`_201641`) underperformed despite higher val MAP, likely due to overfitting.
- **Dynamic Modified Triplet (ce201641)**: Started from `ce_best_201641` checkpoint. Loss = `max(0, margin + 2·d(A,P) − [d(A,N_sim) + d(A,N_rand)])` with margin=0.3, N_sim recomputed every 3 epochs. Trained for 175 epochs (lr=5e-5). Best val MAP=0.8042 at epoch 175, but test MAP=0.6785 — **worse than CE baseline**. The ~12.6 pp val→test gap indicates severe overfitting: val MAP improved monotonically across all 175 epochs while generalization degraded. The model over-adapted to the validation set's N_sim structure rather than learning transferable features. The dynamic N_sim recomputation may have caused the model to "chase" its own validation embeddings, reinforcing in-distribution shortcuts.
