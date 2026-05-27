"""
Uploads PP4 model checkpoints to Hugging Face Hub.
Repo: rojin254/PP4-CQTNet-checkpoints (model repo)
Organizes into subdirs: indian/, indian_vocal/, shs100k/, shs100k_vocal/
"""
import os, sys
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("HF_TOKEN not found in .env")

REPO_ID = "rojin254/PP4-CQTNet-checkpoints"
CKPT_DIR = os.path.join(os.path.dirname(__file__), "check_points")

api = HfApi(token=token)

print(f"Creating model repo: {REPO_ID}")
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
print("Repo ready.\n")

def route(fname):
    f = fname.lower()
    if "indian_vocal" in f:
        return "indian_vocal"
    if "shs100k_vocal" in f:
        return "shs100k_vocal"
    if "shs100k" in f:
        return "shs100k"
    return "indian"

pth_files = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pth")]
print(f"Found {len(pth_files)} checkpoint files.\n")

for fname in sorted(pth_files):
    local = os.path.join(CKPT_DIR, fname)
    dest = f"{route(fname)}/{fname}"
    size_mb = os.path.getsize(local) / 1e6
    print(f"Uploading {dest} ({size_mb:.1f} MB) ...")
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=dest,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"  Done.")

print("\nAll checkpoints uploaded.")
