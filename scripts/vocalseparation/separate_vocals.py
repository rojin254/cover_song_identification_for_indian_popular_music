"""
Vocal Separation Script using Meta's Demucs (htdemucs model).

Separates vocals from audio files for:
  1. Indian test set   (data/test/indian/)
  2. SHS100K train set (data/train/shs100k_train/)
  3. SHS100K test set  (data/test/shs100k-test/)

Output structure mirrors input but under data/{split}_vocal/:
  data/test/indian_vocal/
  data/train/shs100k_train_vocal/
  data/test/shs100k_test_vocal/

Usage:
  python scripts/vocalseparation/separate_vocals.py --dataset indian_test
  python scripts/vocalseparation/separate_vocals.py --dataset shs100k_train
  python scripts/vocalseparation/separate_vocals.py --dataset shs100k_test
  python scripts/vocalseparation/separate_vocals.py --dataset all
"""

import argparse
import os
import sys
import glob
import shutil
import torch
import torchaudio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Demucs imports
from demucs.pretrained import get_model
from demucs.apply import apply_model


# ─── Configuration ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_CONFIG = {
    "indian_test": {
        "input_dir": PROJECT_ROOT / "data" / "test" / "indian",
        "output_dir": PROJECT_ROOT / "data" / "test" / "indian_vocal",
    },
    "shs100k_train": {
        "input_dir": PROJECT_ROOT / "data" / "train" / "shs100k_train",
        "output_dir": PROJECT_ROOT / "data" / "train" / "shs100k_train_vocal",
    },
    "shs100k_test": {
        "input_dir": PROJECT_ROOT / "data" / "test" / "shs100k-test",
        "output_dir": PROJECT_ROOT / "data" / "test" / "shs100k_test_vocal",
    },
}

# ─── Model loading (shared across threads) ──────────────────────────────────

_model = None
_model_lock = Lock()
_device = None


def get_demucs_model(device):
    """Load the htdemucs model once and cache it."""
    global _model, _device
    with _model_lock:
        if _model is None:
            print(f"Loading htdemucs model on {device}...")
            _model = get_model("htdemucs")
            _model.to(device)
            _model.eval()
            _device = device
            print("Model loaded.")
    return _model


# ─── Core separation logic ──────────────────────────────────────────────────

def separate_vocals(input_path: Path, output_path: Path, device: torch.device):
    """
    Separate vocals from a single audio file and save as mp3.
    Uses the htdemucs model (fastest high-quality Demucs v4 model).
    """
    if output_path.exists():
        return f"SKIP (exists): {output_path.name}"

    model = get_demucs_model(device)

    # Load audio
    wav, sr = torchaudio.load(str(input_path))

    # Resample to model's sample rate if needed
    if sr != model.samplerate:
        wav = torchaudio.transforms.Resample(sr, model.samplerate)(wav)
        sr = model.samplerate

    # Ensure stereo
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    # Add batch dimension: (1, channels, samples)
    wav = wav.unsqueeze(0).to(device)

    # Apply model
    with torch.no_grad():
        sources = apply_model(model, wav, device=device, progress=False)

    # sources shape: (1, num_sources, channels, samples)
    # htdemucs sources order: drums, bass, other, vocals
    # Get the index of vocals
    source_names = model.sources
    vocal_idx = source_names.index("vocals")

    vocals = sources[0, vocal_idx].cpu()  # (channels, samples)

    # Save as mp3
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), vocals, sr)

    return f"OK: {output_path.name}"


def process_file(args_tuple):
    """Wrapper for ThreadPoolExecutor."""
    input_path, output_path, device = args_tuple
    try:
        return separate_vocals(input_path, output_path, device)
    except Exception as e:
        return f"ERROR: {input_path.name} - {e}"


# ─── Main pipeline ──────────────────────────────────────────────────────────

def run_separation(dataset_name: str, num_workers: int, device: torch.device):
    config = DATASET_CONFIG[dataset_name]
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all audio files
    extensions = ["*.mp3", "*.wav", "*.flac", "*.ogg"]
    audio_files = []
    for ext in extensions:
        audio_files.extend(input_dir.glob(ext))
    audio_files.sort()

    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return

    # Filter out already-processed files
    tasks = []
    for f in audio_files:
        out_path = output_dir / f.name
        tasks.append((f, out_path, device))

    total = len(tasks)
    already_done = sum(1 for _, out, _ in tasks if out.exists())
    remaining = total - already_done

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Total files: {total} | Already done: {already_done} | Remaining: {remaining}")
    print(f"Workers: {num_workers} | Device: {device}")
    print(f"{'='*60}\n")

    if remaining == 0:
        print("All files already processed. Skipping.")
        return

    # Preload model before threading
    get_demucs_model(device)

    completed = 0
    errors = 0

    if device.type == "cuda":
        # GPU: sequential processing (GPU is the bottleneck, threading won't help)
        for task in tasks:
            result = process_file(task)
            completed += 1
            if result.startswith("ERROR"):
                errors += 1
            if completed % 10 == 0 or completed == remaining:
                print(f"  [{completed}/{total}] {result}")
    else:
        # CPU: use multiple threads for I/O parallelism
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_file, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if result.startswith("ERROR"):
                    errors += 1
                if completed % 10 == 0 or completed == total:
                    print(f"  [{completed}/{total}] {result}")

    print(f"\nDone: {completed - errors} succeeded, {errors} errors out of {total} total.")


def main():
    parser = argparse.ArgumentParser(description="Vocal separation using Demucs htdemucs")
    parser.add_argument(
        "--dataset",
        choices=["indian_test", "shs100k_train", "shs100k_test", "all"],
        required=True,
        help="Which dataset to process",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for CPU mode (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda', 'cpu', or None for auto-detect",
    )
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    datasets = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        run_separation(ds, args.workers, device)

    print("\nAll done!")


if __name__ == "__main__":
    main()
