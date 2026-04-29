# Goal Description

The goal is to update the current cover song detection code to test with the shs100k dataset located in `data/train/shs100k` and `data/test/shs100k`. The dataset has metadata files `data/shs100k_train.csv` and `data/shs100k-test.csv`. The update needs to ensure that:
1. Necessary `.txt` list files are generated for the data loader.
2. The code handles the variable number of cover songs per original song instead of assuming a hardcoded 6 songs per clique.
3. Feature extraction (CQT) from `.mp3` to `.npy` is handled.
4. Training and testing successfully run with the new data.

## User Review Required
None

## Open Questions
- Do you want me to pre-compute all the `.npy` CQT features and save them to a new directory (e.g. `data/cqt_features/`), or should I extract CQT features dynamically in the dataset loader? Pre-computing is faster for training but takes up disk space. (I will write a script to pre-compute them as it's the standard approach in this repository).
- [ANSWER] YES Pre-compute all the `.npy` CQT features and save them to a new directory (e.g. `shs100k_new_pipeline/cqt_features/`)  
- Also make sure to add the generated text files for train and test in `shs100k_new_pipeline/` directory.

## Proposed Changes

### `data`
- Generate `shs100k_new_pipeline/shs100k_train.txt` and `shs100k_new_pipeline/shs100k-test.txt` from the provided CSV files in `data/shs100k-test.csv` and `data/shs100k_train.csv`.

### `cqt_loader.py`
#### [MODIFY] cqt_loader.py
- Update `self.indir` to point to the new `.npy` feature directories.
- Update file reading logic to handle the new id format: e.g., `set_id = filename.split('-')[0]`.
- Update `.txt` filepaths to the newly created `shs100k_train.txt` and `shs100k-test.txt`.
### `main.py`
#### [MODIFY] main.py
- Modify `create_triplets(embeddings, labels)` to remove the hardcoded `if (i+1)%6 == 0:` check. 
- Instead, create a mapping mechanism so that the anchors are always the original song (maybe something like the lowest version value in a clique). For context the existing code is designed for a dataset with exactly 6 covers and   1 original saved in that order But the new dataset has different number of covers per original song


### `gencqt.py` (or a new script)
#### [NEW] generate_cqt.py
- Create a script to extract CQT features from `.mp3` files in `data/train/shs100k_train` and `data/test/shs100k-test` and save them as `.npy` arrays in `shs100k_new_pipeline/cqt_features/` to be consumed by the `SHS100kCQT` dataset loader.

## Verification Plan
1. Run the CQT generation script to convert all `.mp3` files to `.npy`.
2. Run `main.py` to ensure the training loop executes successfully without dimension or index errors.
3. Check the validation and testing steps to ensure `val_slow` correctly calculates the MAP/Rank metrics for cliques with arbitrary sizes.


## Important Note 
- Make sure that any new script files required or any new data files created are all added to a separate folder to avoid confusion and also make a instructions.md file so that other users can recreate the same process