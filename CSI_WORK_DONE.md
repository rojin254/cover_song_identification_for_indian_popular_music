## Objective 
Understand the process of cover song Identification and create a novel approach to implement CSI for Indian popular music. 
Implement existing cover song identification solutions and see how they perform on Indian songs and compare them to the western songs results. 
Pickup the research done previously on this topic and implement the modified triplet loss to improve Indian covers metrics,
Use bigger and diverse Indian dataset to create a more compelling and justifiable result.

## Datasets
For western music we mainly use SHS100K-test dataset. 
We split the same into test and train data (80-20)

For Indian covers I collected a mix of Malayalam songs and Hindi songs and merged it with the existing Hindi song dataset.
Malayalam songs have 2-3 covers avg. per original
Hindi songs have 8-10 covers avg. per original
covers India( existing ) have 1 cover per original.
we combine all theses and split them homogenously with 80-20 split

#### Indian Dataset

| Split | Cliques | Tracks | 
|-------|---------|--------|
| Train | 402 | 2,813 |
| Test | 139 | 709 |
| **Total** | **541** | **3,522** | 


#### SHS100K-test Dataset 

| Split | Cliques | Tracks | 
|-------|---------|--------|
| Train | 92 | 1,826 |
| Test | 24 | 591 |
| **Total** | **119** | **2,417** |

---
### Baseline models
1. CQTNET
2. TPPNET
3. Byte-Cover
### Experiments Performed so far
#### Base Dataset:
1. Run inference on both test splits of shs100k and Indian covers for all 3 datasets with the base checkpoint as the one provided by them which was trained on western covers
2. Run finetuning by training the base checkpoint along with our train data to get the finetuned checkpoint and the test all 3 models ( byte cover finetuning takes too long)
3. Compare the results of Indian cover songs with western covers
#### Vocally Separated Dataset.
1. Separate the vocals for all splits in both datasets using tools like DEMUCS from meta
2. Perform testing on base checkpoints with this new dataset
3. Finetune and check the results again with new dataset.
4. compare normal covers results with vocal separated covers result.

#### Implementing Modified Triplet Loss
1. using Timothy's code we can implement the testing and finetuning of CQTNET, Triplet Loss finetuning and Modified Triplet loss finetuning.
2. CQT will be loaded with base western music trained checkpoint, and then finetuned with our train split.
3. Then on top of that we finetune with triplet loss finetuning by taking positive and negative pairs. 
4. Then we experiment with modified triplet loss on top of finetuned CQTNET. 
	1. Modified Triplet loss uses positive pair, negative pair, and most positive negative pair (Nsim)
	2. Choosing Nsim can be done based on finetuned CQT model to create embeddings and compare the distance between embeddings and find a negative pair which has the lowest distance

#### Code running instructions

CE fine-tune from western checkpoint only:**
```bash
cd indian_new_pipeline
python main.py transfer_learning
```

CE fine-tune, then Triplet Loss fine-tune:**
```bash
python main.py transfer_learning --fine_tune=True
```

Full pipeline: CE → Triplet → Modified Triplet (pre-computed N_sim via `nsim_lookup.json`):**
```bash
# Requires nsim_lookup.json (run compute_nsim_lookup.py first if missing)
python main.py transfer_learning --modified_triplet=True
```

Dynamic Modified Triplet (N_sim derived from model embeddings, no lookup file needed):**
```bash
python main.py transfer_learning --dynamic_triplet=True
```

**Skip CE stage, run dynamic triplet from an existing checkpoint:**
```bash
python main.py transfer_learning --dynamic_triplet=True --skip_transfer=True  --load_model_path=../check_points/best_abc.pth
```

---

**How N_sim is computed in the dynamic variant**,
this is main.py:
1. Runs a full pass over the training set to get all embeddings from the current model
2. L2-normalises them and computes a cosine similarity matrix (on CPU to avoid OOM)
3. For each original track, masks same-clique songs and picks `argmax` similarity, the hardest negative from a different clique
4. This is refreshed every `nsim_refresh_interval=3` epochs so N_sim tracks the evolving embedding space



### Pending Work
- Evaluation of HuBERT and MERT
- Train CQTNet from scratch with indian covers