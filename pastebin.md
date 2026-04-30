
```
python scripts/vocalseparation/separate_vocals.py --dataset all
```
```
Based on the paper's results and your current progress, **vocal separation is clearly the highest-priority next step**, not modified triplet loss yet.

**Why vocal separation first:**
- It's the single biggest gain in the paper: baseline MAP jumps from **0.7238 → 0.8629** (vocal sep alone), while triplet loss only moved it from ~0.73 to ~0.77
- Modified triplet loss only outperforms standard triplet loss meaningfully *when combined with vocal-separated data* (0.9234 vs 0.9079)
- The paper's best result is **Mod Trip.Loss + DA + VocalSep = 0.9234** — vocal sep is a prerequisite for that pipeline

**Recommended progression:**
1. **Vocal sep + fine-tuned CQTNet** (expected ~0.90 MAP)
2. **DA + Vocal sep** (expected ~0.91 MAP)
3. **Triplet loss + DA + Vocal sep**
4. **Modified triplet loss + DA + Vocal sep** (expected ~0.92 MAP)

**On test data:** Use vocal-separated test data when evaluating models trained on vocal-separated features. The pipeline must be consistent — a model that learned vocal-only CQT features will produce misleading embeddings if evaluated on full-mix CQT. The paper confirms this implicitly (all vocal sep rows use vocal sep for both train and test).

**On N_sim (for when you get there):** It's computed per-epoch before the training loop:
1. Run the current model in eval mode over all training songs to get embeddings
2. For each anchor `A_orig`, find the non-cover song embedding that minimizes cosine distance to `A_orig` — that's `N_sim`
3. Store these as a lookup table, then use them in the loss during that epoch's training

So focus on getting the vocal-separated CQT features generated first — that's the bottleneck gating everything downstream.
```
# main Task
Now based on your Idea create a plan for me considering I have all the required dataests now (gnore data augmentation for now.), I need to make some base level things clear.
1. @file:cqtnet_indian_best.pth and @file:cqtnet_shs100k_best.pth  are the base weights over which cqtnet was finetuned. SO consider them the best weights after cqtnet training
2. @file:CQTNet_SHS100k_finetune_epoch_74.pth and @file:CQTNet_transfer_learning_epoch_64.pth are the best models for shs100k and indain covers after normal triplet loss finetuning. 

From now on Only do triplet loss and other finetuning steps and dont do a base CQTNet finetuning and instead use the existing best weights (point 1).

For shs100k vocal and indian vocal create two separate scripts in different directories similar to indian_new_pipeline and shs100k_new_pipeline. Also save any required .npy files in the subfolders with same structure as well.

When creating new scripts make sure to take care of issues like correct anchor point setting based on mapping as already implemented in previous pipelines