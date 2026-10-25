import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..artifacts import sha256, write_json
from .cqt import extract, recipe
from .dataset import tonic_key
from .manifests import load_partition, prepare, read_inventory, write_csv
from .tonic import ESTIMATOR, estimate_tonic

ASSETS = {
    "dataset": ("rojin254/PP4-indian-music-csi", "dataset", "973478c5b13ee5dcd153222289d867a4cc182abd"),
    "checkpoints": ("rojin254/PP4-CQTNet-checkpoints", "model", "5ae4cbfb60fb798c0c5c1854503f23bc543bc7f8"),
}


def fetch(asset, filenames, destination):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    repo, repo_type, revision = ASSETS[asset]
    if not filenames:
        raise ValueError("Explicit --file selection is required")
    for filename in filenames:
        path = Path(filename)
        if path.is_absolute() or ".." in path.parts or any(c in filename for c in "*?[]"):
            raise ValueError("Fetch accepts exact relative filenames, never patterns")
    destination = Path(destination).resolve()
    files, failures = [], []
    destination.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        try:
            path = hf_hub_download(repo_id=repo, repo_type=repo_type, revision=revision, filename=filename, local_dir=destination)
        except HfHubHTTPError as exc:
            failures.append({"file": filename, "error": str(exc)})
            continue
        files.append({"file": filename, "sha256": sha256(path), "bytes": Path(path).stat().st_size})
    result = {"repository": repo, "revision": revision, "requested": filenames, "downloaded": files, "failures": failures}
    # Different subset fetches have distinct ledgers, so evidence is never overwritten.
    import hashlib
    key = hashlib.sha256(json.dumps(sorted(filenames)).encode()).hexdigest()[:16]
    write_json(destination / f"fetch-{asset}-{key}.json", result)
    if failures:
        raise RuntimeError(f"Asset fetch incomplete: {len(failures)} failed; see fetch-{asset}-{key}.json")
    return result


def feature_job(job):
    row, root, settings = job
    root = Path(root)
    audio = root / row["audio_path"]
    audio_hash = sha256(audio)
    fingerprint, spec = recipe(settings, audio_hash)
    relative = Path("features") / row["dataset"] / row["modality"] / fingerprint / f"{row['track_id']}.npy"
    path = root / relative
    sidecar = path.with_suffix(".json")
    if path.exists():
        if not sidecar.exists():
            raise ValueError(f"Feature has no provenance: {path}")
        previous = json.loads(sidecar.read_text())
        if previous["sha256"] != sha256(path) or previous["recipe"] != spec:
            raise ValueError(f"Corrupt/stale existing feature: {path}")
    else:
        features = extract(audio, settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, features)
        temporary.replace(path)
        write_json(sidecar, {"recipe": spec, "recipe_hash": fingerprint, "sha256": sha256(path), "dtype": str(features.dtype), "shape": list(features.shape)})
    return {**row, "feature_path": str(relative), "feature_sha256": sha256(path), "feature_recipe": fingerprint}


def prepare_data(inventory, config, features=False, workers=0):
    root = Path(config["paths"]["data"])
    rows = read_inventory(inventory)
    if features:
        jobs = [(row, str(root), config["preprocess"]) for row in rows]
        if workers:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                rows = list(executor.map(feature_job, jobs))
        else:
            rows = [feature_job(job) for job in jobs]
        prepared = root / "metadata" / "prepared_tracks.csv"
        write_csv(prepared, rows)
        inventory = prepared
    return prepare(inventory, root, config["split"])


def build_tonic(config, output, check=False):
    entries = {}
    sets = [(config["dataset"], config["modality"], p) for p in ("train", "val")]
    sets.extend((*name.split("."), "test") for name in config["evaluation"]["sets"])
    for dataset, modality, partition in sets:
        for row in load_partition(config, partition, dataset, modality):
            features = np.load(Path(config["paths"]["data"]) / row["feature_path"], allow_pickle=False)
            entries[tonic_key(row)] = {
                "tonic": estimate_tonic(features), "feature_sha256": row["feature_sha256"],
                "feature_recipe": row["feature_recipe"], "method": ESTIMATOR,
            }
    if check:
        if json.loads(Path(output).read_text()) != entries:
            raise ValueError("Tonic index is missing entries, stale or has unexpected membership")
    else:
        write_json(output, entries)
    return {"tonic_entries": len(entries), "path": str(Path(output).resolve()), "checked": check}


def separate_vocals(inputs, output, model, device, allow_model_download=False):
    import importlib.util
    if importlib.util.find_spec("demucs") is None:
        raise ValueError("Optional Demucs is not installed; install it explicitly on the authorized remote host")
    if not allow_model_download:
        raise ValueError("Demucs may download weights; explicitly pass --allow-model-download to authorize it")
    for path in inputs:
        if not Path(path).is_file():
            raise ValueError(f"Missing separation input: {path}")
    command = [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", model, "-d", device, "-o", output, *inputs]
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise RuntimeError(f"Demucs exited with status {result.returncode}")
    products = []
    for source in inputs:
        vocal = Path(output) / model / Path(source).stem / "vocals.wav"
        products.append({"input": str(Path(source).resolve()), "input_sha256": sha256(source), "output": str(vocal.resolve()), "sha256": sha256(vocal)})
    write_json(Path(output) / "separation.json", {"model": model, "device": device, "files": products})
    return {"outputs": products}
