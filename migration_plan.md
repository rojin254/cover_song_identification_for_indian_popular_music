# PP4 Migration Plan

## Context

PP4 is a 34 GB research ML codebase (Cover Song Identification for Indian Music, ISMIR 2025) currently on a Lightning AI teamspace studio. The goal is to migrate all components — code, 30 GB of audio data, 317 MB of model checkpoints, documentation — to persistent storage that survives beyond the studio, is accessible from a new system, and is organized for efficient transfer. The migration must be planned quantitatively because the assets span 4 orders of magnitude in size and regeneration feasibility varies per component.

---

## Inventory (Verified Sizes)

| Component | Size | Files | Action |
|---|---|---|---|
| Audio: data/train/indian | 5.7 GB | 2,809 MP3 | Upload to HF Datasets |
| Audio: data/train/indian_vocal_train | 4.9 GB | 2,813 MP3 | Upload to HF Datasets |
| Audio: data/train/shs100k_train | 6.7 GB | 1,826 MP3 | Upload to HF Datasets |
| Audio: data/train/shs100k_train_vocal | 5.7 GB | 1,826 MP3 | Upload to HF Datasets |
| Audio: data/test/* (4 dirs) | 6.6 GB | 2,600 MP3 | Upload to HF Datasets |
| **Audio total** | **~30 GB** | **11,874** | HF Datasets |
| CQT features (4 pipeline dirs) | ~3.8 GB | 11,870 .npy | **Regenerate** on new system (2–4 hrs) |
| Model checkpoints | 317 MB | 16 .pth (~20 MB each) | HF Hub model repo |
| Large training logs | ~8.8 MB | ~6 .log files | Compress + HF Hub |
| Python source + JSON logs + docs | ~600 KB | ~70 files | **Git** (commit + push) |

---

## Part 1 — Storage Strategy

### Code → GitHub (existing remote: https://github.com/Timothy-John/PP4.git)

All 5 branches (master, CQTNet, new-indian-dataset-tests, MERT_HuBERT, encodec-mel) must be preserved. Each documents a distinct experimental approach.

**Currently untracked files to commit before migration:**
- 4× `*/main.py` (modified with dynamic triplet loss)
- 4× `*/run_all_*.sh`, `shs100k_vocal_pipeline/run_from_ce.sh`
- All `*/logs/` directories (JSON files only — not the multi-MB `.log` files)
- All `*/results.md`, `*/instructions.md`
- Documentation: `EXPERIMENT_AND_RESULTS_REPORT.md`, `ISMIR_CSI_PAPER_CONTENT.md`, `final_report.tex`, `interim_report.tex`, `Pre-val-report.tex`, `quick_result.md`, `report.md`, `report2.md`, `image.png`
- Scripts: `kill_gpu_procs.sh`, `test_vocal_on_indian.py`

**Do NOT commit:** `*.pth` checkpoints, `*.npy` CQT features, `*.mp3` audio, large `*.log` files.

### Checkpoints → HF Hub model repo

Repo: `Timothy-John/PP4-CQTNet-checkpoints`

Organize by pipeline:
```
indian/        ← cqtnet_indian_best.pth, CQTNet_indian_dyn_triplet_*.pth
indian_vocal/  ← CQTNet_indian_vocal_ce_*.pth, CQTNet_indian_vocal_*triplet*.pth
shs100k/       ← cqtnet_shs100k_best.pth, CQTNet_shs100k_*.pth
shs100k_vocal/ ← CQTNet_shs100k_vocal_*.pth
logs/          ← pp4_large_logs.tar.gz
```

14 of 16 checkpoints are currently **untracked in git** — these must be uploaded to HF Hub before the studio is decommissioned. The 2 tracked ones (`cqtnet_indian_best.pth`, `cqtnet_shs100k_best.pth`) are already in git history but should also be on HF Hub for easier access.

### Audio → HF Datasets

Repo: `Timothy-John/PP4-indian-music-csi`

Upload the entire `data/` directory preserving the subdirectory tree. Use `upload_folder` with `multi_commits=True` for the 30 GB payload. The metadata CSV/TXT files in `data/` are included automatically.

### CQT features (~3.8 GB) — Do NOT store, regenerate

MP3→CQT extraction takes 2–4 hours on any modern CPU using the existing multiprocessing scripts. This is faster than the time it would take to upload and download 3.8 GB. `prepare_data.py` and `generate_cqt.py` are idempotent (skip existing `.npy` files).

---

## Part 2 — Code Consolidation (Optional but Recommended)

The 4 pipeline `main.py` files are 875–901 lines each and differ by fewer than 10 semantic lines. The 4 `cqt_loader.py` files differ by only dataset-specific class names and one metadata parsing difference. This duplication is the main maintenance burden.

### Proposed structure

**New: `pipelines/train.py`** — unified training script  
Accept `--dataset_type=indian|shs100k` and `--audio_type=source|vocal` as top-level arguments. The 5 lines that differ between pipelines resolve to dict lookups on these two params.

**New: `pipelines/cqt_loader.py`** — unified dataset class  
Single `CQTDataset(dataset_type, mode)` parameterized by a `DATASET_CONFIG` dict. The only real difference between Indian and SHS100K loaders is:
- `label_attr`: Indian uses `.label_count` + `.label_map`; SHS100K uses `.set` + `.set_dict`
- `is_original` detection: Indian uses filename suffix (`-o`); SHS100K uses third CSV column (`== '1'`)

**Keep pipeline-specific thin shims:** Replace each `*/main.py` with a 5-line shim that sets `DATASET_TYPE` and `AUDIO_TYPE` then imports from `pipelines.train`. This makes all existing shell scripts continue to work unchanged.

**Files to change:**

| Action | File |
|---|---|
| Create | `pipelines/__init__.py` |
| Create | `pipelines/train.py` (merged from 4 × `*/main.py`) |
| Create | `pipelines/cqt_loader.py` (merged from 4 × `*/cqt_loader.py`) |
| Replace | `indian_new_pipeline/main.py` → 5-line shim |
| Replace | `indian_vocal_pipeline/main.py` → 5-line shim |
| Replace | `shs100k_new_pipeline/main.py` → 5-line shim |
| Replace | `shs100k_vocal_pipeline/main.py` → 5-line shim |
| Keep | All `*/run_all_*.sh` (unchanged) |
| Keep | `models/`, `config.py`, `utility.py`, `scripts/` (unchanged) |

This consolidation reduces the codebase from ~3,600 lines of duplicated pipeline code to ~1,100 lines in `pipelines/train.py` + `pipelines/cqt_loader.py`.

---

## Part 3 — Migration Steps

### Step 1 — Commit all unstaged code (15 min)

```bash
cd /teamspace/studios/this_studio/PP4

git add \
  "ISMIR_2025_cover_song (1).md" ISMIR_CSI_PAPER_CONTENT.md \
  EXPERIMENT_AND_RESULTS_REPORT.md Pre-val-report.tex final_report.tex \
  interim_report.tex image.png kill_gpu_procs.sh test_vocal_on_indian.py \
  quick_result.md report.md report2.md \
  indian_new_pipeline/main.py indian_new_pipeline/run_all_indian.sh \
  indian_new_pipeline/results.md indian_new_pipeline/logs/ \
  indian_vocal_pipeline/main.py indian_vocal_pipeline/results.md \
  indian_vocal_pipeline/logs/ \
  shs100k_new_pipeline/main.py shs100k_new_pipeline/run_all_shs.sh \
  shs100k_new_pipeline/logs/ \
  shs100k_vocal_pipeline/main.py shs100k_vocal_pipeline/run_all_shs_vocal.sh \
  shs100k_vocal_pipeline/run_from_ce.sh shs100k_vocal_pipeline/logs/

git commit -m "Add all untracked pipeline code, reports, and logs before migration"
git push origin new-indian-dataset-tests
```

### Step 2 — Upload checkpoints to HF Hub (20–30 min, ~317 MB)

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import HfApi, login
login()
api = HfApi()
api.create_repo(repo_id="Timothy-John/PP4-CQTNet-checkpoints", repo_type="model", exist_ok=True)

import os
CKPT_DIR = "/teamspace/studios/this_studio/PP4/check_points"

routing = {
    "indian_vocal": "indian_vocal",
    "shs100k_vocal": "shs100k_vocal",
    "shs100k": "shs100k",
}

for fname in os.listdir(CKPT_DIR):
    if not fname.endswith(".pth"):
        continue
    dest_dir = "indian"
    for key, folder in routing.items():
        if key in fname:
            dest_dir = folder
            break
    api.upload_file(
        path_or_fileobj=os.path.join(CKPT_DIR, fname),
        path_in_repo=f"{dest_dir}/{fname}",
        repo_id="Timothy-John/PP4-CQTNet-checkpoints",
        repo_type="model",
    )
    print(f"Uploaded: {fname}")
EOF
```

### Step 3 — Compress and upload large logs (5 min)

```bash
find /teamspace/studios/this_studio/PP4 -name "*.log" -path "*/logs/*" \
  | tar -czf /tmp/pp4_large_logs.tar.gz -T -
# Expected: ~8.8 MB → ~1.5 MB after compression

python -c "
from huggingface_hub import HfApi
HfApi().upload_file(
    path_or_fileobj='/tmp/pp4_large_logs.tar.gz',
    path_in_repo='logs/pp4_large_logs.tar.gz',
    repo_id='Timothy-John/PP4-CQTNet-checkpoints',
    repo_type='model'
)
print('Logs uploaded')
"
```

### Step 4 — Upload audio data to HF Datasets (1.5–7 hrs, 30 GB)

Run in a tmux session — this is the long-pole task.

```bash
huggingface-cli login

python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id="Timothy-John/PP4-indian-music-csi", repo_type="dataset", exist_ok=True)
api.upload_folder(
    folder_path="/teamspace/studios/this_studio/PP4/data",
    repo_id="Timothy-John/PP4-indian-music-csi",
    repo_type="dataset",
    multi_commits=True,
    multi_commits_verbose=True,
    ignore_patterns=["*.npy", "*.pyc", "__pycache__"],
)
print("Audio upload complete")
EOF
```

Verify upload (run after completion):
```bash
python -c "
from huggingface_hub import list_repo_tree
files = list(list_repo_tree('Timothy-John/PP4-indian-music-csi', repo_type='dataset', recursive=True))
mp3s = [f for f in files if str(f).endswith('.mp3')]
print(f'Total MP3s on HF: {len(mp3s)}')  # expected: 11874
"
```

---

## Part 4 — New System Reconstruction

### 4.1 Environment

```bash
conda create -n pp4 python=3.12 && conda activate pp4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install librosa tqdm numpy pandas fire demucs huggingface_hub
```

### 4.2 Clone code

```bash
git clone https://github.com/Timothy-John/PP4.git
cd PP4
git checkout new-indian-dataset-tests
```

### 4.3 Download checkpoints (~5 min, 317 MB)

```bash
python - <<'EOF'
from huggingface_hub import hf_hub_download, list_repo_files
import os, shutil
REPO = "Timothy-John/PP4-CQTNet-checkpoints"
os.makedirs("check_points", exist_ok=True)
for f in list_repo_files(REPO, repo_type="model"):
    if f.endswith(".pth"):
        src = hf_hub_download(repo_id=REPO, filename=f, repo_type="model")
        shutil.copy(src, f"check_points/{os.path.basename(f)}")
        print(f"Downloaded: {f}")
EOF
```

### 4.4 Download audio data (~1.5–6 hrs, 30 GB)

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
import shutil, os
local = snapshot_download(
    repo_id="Timothy-John/PP4-indian-music-csi",
    repo_type="dataset",
    local_dir="./hf_data_cache",
)
shutil.copytree(os.path.join(local, "train"), "data/train", dirs_exist_ok=True)
shutil.copytree(os.path.join(local, "test"),  "data/test",  dirs_exist_ok=True)
EOF
```

Verify:
```bash
find data/train/indian -name "*.mp3" | wc -l         # expected: 2809
find data/train/shs100k_train -name "*.mp3" | wc -l  # expected: 1826
find data/test/indian -name "*.mp3" | wc -l           # expected: 709
find data/test/shs100k-test -name "*.mp3" | wc -l    # expected: 591
```

### 4.5 Regenerate CQT features (2–4 hrs, unattended)

```bash
cd indian_new_pipeline    && python prepare_data.py && cd ..
cd indian_vocal_pipeline  && python prepare_data.py && cd ..
cd shs100k_new_pipeline   && python generate_cqt.py && cd ..
cd shs100k_vocal_pipeline && python prepare_data.py && cd ..
```

### 4.6 Verify end-to-end

```bash
# Quick model load check
python - <<'EOF'
import torch
from models import CQTNet
from torch import nn
for path, n in [("check_points/cqtnet_indian_best.pth", 120),
                ("check_points/CQTNet_shs100k_mod_triplet_best_shs100k_20260502_184211.pth", 10000)]:
    m = CQTNet(); m.fc1 = nn.Linear(300, n)
    m.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
    _, emb = m(torch.zeros(1, 1, 84, 400))
    print(f"OK {path} — emb: {emb.shape}")
EOF

# Quick evaluation on best Indian model
cd indian_new_pipeline
python main.py evaluate \
  --checkpoint=../check_points/CQTNet_indian_dyn_triplet_best_indian_fin_map76.pth
# Expected: MAP ~0.76
```

---

## Timeline Summary

| Task | Wall-clock | Active time | Notes |
|---|---|---|---|
| Commit + push code | 5 min | 5 min | Do first |
| Upload 16 checkpoints | 20–30 min | 5 min | Run in background |
| Compress + upload logs | 5 min | 2 min | — |
| Upload 30 GB audio | 1.5–7 hrs | 5 min | Start in tmux, let run overnight |
| Clone + download checkpoints (new system) | 10 min | 5 min | — |
| Download audio (new system) | 1.5–6 hrs | 5 min | Start in tmux |
| Regenerate CQT features (new system) | 2–4 hrs | 0 min | Unattended, runs in parallel with download |
| Verify | 10 min | 10 min | — |

**Total human active time: ~37 minutes. Total wall-clock: 4–12 hours (dominated by data upload/download and CQT regeneration, all running unattended).**
