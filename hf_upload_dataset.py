"""
Uploads the PP4 audio dataset to Hugging Face Datasets using upload_large_folder.
- Resumable: progress saved in data/.cache/.huggingface/
- Parallel workers: uses half of available CPU cores
- Handles LFS automatically
Repo: rojin254/PP4-indian-music-csi
"""
import os, sys
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("HF_TOKEN not found in .env")

REPO_ID = "rojin254/PP4-indian-music-csi"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

api = HfApi(token=token)

print(f"Creating dataset repo: {REPO_ID}")
api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
print("Repo ready. Starting upload_large_folder ...")
print(f"Folder: {DATA_DIR}")
print("Progress report every 60s. Resume-safe: re-run this script if interrupted.\n")

api.upload_large_folder(
    repo_id=REPO_ID,
    folder_path=DATA_DIR,
    repo_type="dataset",
    ignore_patterns=["*.npy", "*.pyc", "__pycache__", ".cache"],
    print_report=True,
    print_report_every=60,
)

print("\nAll uploads complete.")
