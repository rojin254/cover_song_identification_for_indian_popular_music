"""
Sample inference using HF datasets + HF Hub.

Downloads:
  - 10 songs from rojin254/PP4-indian-music-csi  (streamed, ~20 MB total)
  - checkpoint  from rojin254/PP4-CQTNet-checkpoints  (~21 MB)

Usage:
    python hf_inference_sample.py
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import librosa
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from models import CQTNet
from utility import norm, calc_MAP

# ── config ────────────────────────────────────────────────────────────────────
CHECKPOINT_REPO = "rojin254/PP4-CQTNet-checkpoints"
CHECKPOINT_FILE = "indian_vocal/CQTNet_indian_vocal_dyn_triplet_best_vocal_dyn_triplet_ce201641_full.pth"
DATASET_REPO    = "rojin254/PP4-indian-music-csi"
AUDIO_PATTERN   = "test/indian_vocal/*.mp3"    # glob within the HF repo
TRAIN_TXT       = os.path.join(ROOT, "indian_vocal_pipeline", "indian_train.txt")
TEST_TXT        = os.path.join(ROOT, "indian_vocal_pipeline", "indian_test.txt")
N_SAMPLES       = 10                           # songs to pull from HF
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── helpers ───────────────────────────────────────────────────────────────────

def get_num_classes(train_txt):
    cliques = set()
    with open(train_txt) as f:
        for line in f:
            if line.strip():
                cliques.add(line.strip().split(',')[1])
    return len(cliques)


def build_clique_map(test_txt):
    """Returns {filename_stem: clique_int}."""
    mapping = {}
    with open(test_txt) as f:
        for line in f:
            if line.strip():
                stem, clique = line.strip().split(',')
                mapping[stem] = int(clique)
    return mapping


def audio_array_to_cqt(array, sr):
    """Decoded audio array → CQT (84, T) using same params as prepare_data.py."""
    y = array.astype(np.float64)
    if y.ndim > 1:          # stereo → mono
        y = y.mean(axis=1)
    cqt = np.abs(librosa.cqt(y=y, sr=sr))
    mean_size = 20
    n_frames = cqt.shape[1] // mean_size
    if n_frames == 0:
        return None
    new_cqt = np.zeros((cqt.shape[0], n_frames), dtype=np.float64)
    for i in range(n_frames):
        new_cqt[:, i] = cqt[:, i * mean_size:(i + 1) * mean_size].mean(axis=1)
    return new_cqt


def pad_or_truncate(cqt, target_length=400, target_freq=84):
    """Normalise + shape to (1, 84, 400) tensor."""
    cqt = cqt.astype(np.float32)
    cqt = cqt / (np.max(np.abs(cqt)) + 1e-6)

    if cqt.shape[0] > target_freq:
        cqt = cqt[:target_freq, :]
    elif cqt.shape[0] < target_freq:
        cqt = np.pad(cqt, ((0, target_freq - cqt.shape[0]), (0, 0)))

    if cqt.shape[1] < 200:
        cqt = np.pad(cqt, ((0, 0), (0, 200 - cqt.shape[1])))
    if cqt.shape[1] > target_length:
        cqt = cqt[:, :target_length]
    elif cqt.shape[1] < target_length:
        cqt = np.pad(cqt, ((0, 0), (0, target_length - cqt.shape[1])))

    return torch.tensor(cqt).unsqueeze(0)   # (1, 84, 400)


def load_model(checkpoint_path, num_classes):
    model = CQTNet()
    model.fc1 = nn.Linear(300, num_classes)
    state = torch.load(checkpoint_path, map_location="cpu")
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    print(f"  Loaded {len(filtered)}/{len(model_state)} weight tensors")
    return model.to(DEVICE).eval()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    load_dotenv(os.path.join(ROOT, ".env"))
    token = os.environ.get("HF_TOKEN")
    print(f"Device : {DEVICE}")
    print(f"Samples: {N_SAMPLES}\n")

    # 1. Download checkpoint from HF Hub
    print("[1/4] Downloading checkpoint ...")
    ckpt_path = hf_hub_download(
        repo_id=CHECKPOINT_REPO,
        filename=CHECKPOINT_FILE,
        repo_type="model",
        token=token,
    )
    print(f"  {CHECKPOINT_FILE}\n")

    # 2. Stream N_SAMPLES audio files from HF Datasets
    #    audiofolder + hf:// streams files on demand — no full download
    print(f"[2/4] Streaming {N_SAMPLES} songs from HF Datasets ...")
    ds = load_dataset(
        "audiofolder",
        data_files={"test": f"hf://datasets/{DATASET_REPO}/{AUDIO_PATTERN}"},
        split="test",
        streaming=True,
        token=token,
    )
    samples = list(ds.take(N_SAMPLES))
    print(f"  Fetched {len(samples)} samples\n")

    # 3. Load model
    print("[3/4] Loading model ...")
    num_classes = get_num_classes(TRAIN_TXT)
    print(f"  num_classes: {num_classes}")
    model = load_model(ckpt_path, num_classes)
    print()

    # 4. Run inference
    print("[4/4] CQT + inference ...")
    clique_map = build_clique_map(TEST_TXT)

    embeddings_list, labels_list, skipped = [], [], 0

    for sample in tqdm(samples, desc="Songs"):
        # datasets gives audio as {"array": ndarray, "sampling_rate": int, "path": str}
        audio    = sample["audio"]
        # extract filename stem from path, e.g. ".../114-1-ci-c.mp3" → "114-1-ci-c"
        stem     = os.path.splitext(os.path.basename(audio["path"]))[0]
        clique   = clique_map.get(stem)

        if clique is None:
            tqdm.write(f"  WARNING: no clique for '{stem}', skipping")
            skipped += 1
            continue

        cqt = audio_array_to_cqt(audio["array"], audio["sampling_rate"])
        if cqt is None:
            skipped += 1
            continue

        tensor = pad_or_truncate(cqt).unsqueeze(0).to(DEVICE)   # (1, 1, 84, 400)
        with torch.no_grad():
            _, emb = model(tensor)

        embeddings_list.append(emb.cpu().numpy())
        labels_list.append(clique)

    if skipped:
        print(f"  Skipped {skipped} files")

    # 5. Metrics
    embeddings = np.concatenate(embeddings_list, axis=0)   # (N, 300)
    labels     = np.array(labels_list)
    embeddings = norm(embeddings)
    dis2d      = -np.matmul(embeddings, embeddings.T)      # cosine distance

    map_score, top10, rank1 = calc_MAP(dis2d, labels)

    print("\n" + "=" * 50)
    print(f"  Checkpoint : {os.path.basename(CHECKPOINT_FILE)}")
    print(f"  Songs      : {len(labels)}  (subset of 709 test songs)")
    print(f"  MAP        : {map_score:.4f}")
    print(f"  Top10      : {top10:.4f}")
    print(f"  Rank1      : {rank1:.2f}")
    print("=" * 50)
    print("\nNote: metrics on 10 songs are illustrative only.")
    print("Run on full test set (N_SAMPLES=709) for published results.")


if __name__ == "__main__":
    main()
