"""Small synthetic CQT inventory for lifecycle checks, never a benchmark."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ..artifacts import sha256, write_json
from .manifests import prepare, write_csv


def create_fixture(output):
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Fixture output must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    rows, rng = [], np.random.default_rng(42)
    spec = {"synthetic": True, "seed": 42, "shape": [84, 200], "dtype": "float64", "recipe": "uniform_fixture_v1"}
    recipe_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    for dataset in ("indian", "shs100k"):
        for modality in ("mix", "vocal"):
            for clique, partition in ((0, "train"), (1, "train"), (2, "val"), (3, "test")):
                for version in range(2):
                    track = f"{clique}-{version}"
                    relative = Path("features") / dataset / modality / recipe_hash / f"{track}.npy"
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    values = rng.random((84, 200))
                    np.save(path, values)
                    checksum = sha256(path)
                    write_json(path.with_suffix(".json"), {"recipe": spec, "recipe_hash": recipe_hash, "sha256": checksum, "dtype": "float64", "shape": [84, 200]})
                    rows.append({
                        "dataset": dataset, "modality": modality, "track_id": track, "clique_id": str(clique),
                        "source_track_id": track, "is_original": "true" if version == 0 else "false",
                        "audio_path": f"synthetic-no-audio/{dataset}/{modality}/{track}.wav",
                        "partition": partition, "feature_path": str(relative), "feature_sha256": checksum,
                        "feature_recipe": recipe_hash,
                    })
    inventory = root / "metadata" / "tracks.csv"
    write_csv(inventory, rows)
    write_json(root / "fixture.json", spec)
    result = prepare(inventory, root, {"validation_fraction": 0.1, "seed": 42, "rounding": "ceil"})
    return {**result, "root": str(root), "synthetic": True}
