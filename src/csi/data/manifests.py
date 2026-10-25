import csv
import json
import math
import random
from pathlib import Path

import numpy as np

from ..artifacts import sha256, write_json

REQUIRED = ("dataset", "modality", "track_id", "clique_id", "is_original", "source_track_id", "audio_path", "partition")
FEATURES = ("feature_path", "feature_sha256", "feature_recipe")


def read_inventory(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(REQUIRED).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing inventory columns: {REQUIRED}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty inventory: {path}")
    return validate_inventory(rows)


def validate_inventory(rows):
    seen, clique_partitions, family_partitions, identities = set(), {}, {}, {}
    for row in rows:
        if any(not row.get(key) for key in REQUIRED):
            raise ValueError("Inventory has empty required fields")
        if row["dataset"] not in ("indian", "shs100k") or row["modality"] not in ("mix", "vocal"):
            raise ValueError("Invalid dataset/modality")
        if row["partition"] not in ("train", "val", "test") or row["is_original"] not in ("true", "false"):
            raise ValueError("Invalid partition or is_original; use train/val/test and true/false")
        for field in ("track_id", "clique_id"):
            if "/" in row[field] or "\\" in row[field] or row[field] in (".", ".."):
                raise ValueError(f"Unsafe {field}: {row[field]}")
        identity = (row["dataset"], row["modality"], row["track_id"])
        if identity in seen:
            raise ValueError(f"Duplicate track identity: {identity}")
        seen.add(identity)
        key = (row["dataset"], row["track_id"])
        meaning = (row["clique_id"], row["source_track_id"], row["partition"], row["is_original"])
        if key in identities and identities[key] != meaning:
            raise ValueError(f"Cross-modality identity mismatch: {key}")
        identities[key] = meaning
        for mapping, group in (
            (clique_partitions, (row["dataset"], row["clique_id"])),
            (family_partitions, (row["dataset"], row["source_track_id"])),
        ):
            previous = mapping.setdefault(group, row["partition"])
            if previous != row["partition"]:
                raise ValueError(f"Clique or augmentation family crosses partitions: {group}")
        for field in ("audio_path", "feature_path"):
            path = Path(row.get(field, ""))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{field} must be relative to the data root")
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(REQUIRED + FEATURES)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare(inventory, root, split):
    rows = read_inventory(inventory)
    root = Path(root).resolve()
    output = root / "manifests"
    for dataset in sorted({r["dataset"] for r in rows}):
        selected = [dict(r) for r in rows if r["dataset"] == dataset]
        train_cliques = sorted({r["clique_id"] for r in selected if r["partition"] == "train"})
        explicit_val = {r["clique_id"] for r in selected if r["partition"] == "val"}
        shuffled = list(train_cliques)
        random.Random(split["seed"]).shuffle(shuffled)
        count = math.ceil(len(shuffled) * split["validation_fraction"])
        validation = explicit_val or set(shuffled[:count])
        membership = {
            "inventory_sha256": sha256(inventory), "split": split,
            "validation_cliques": sorted(validation), "base_train_order": train_cliques,
            "test_tracks": sorted({r["track_id"] for r in selected if r["partition"] == "test"}),
        }
        destination = output / dataset
        saved = destination / "split.json"
        for row in selected:
            if row["partition"] == "train" and row["clique_id"] in validation:
                row["partition"] = "val"
        membership["memberships"] = {
            part: sorted(f"{r['modality']}.{r['track_id']}" for r in selected if r["partition"] == part)
            for part in ("train", "val", "test")
        }
        if saved.exists() and json.loads(saved.read_text()) != membership:
            raise ValueError(f"Persisted split differs; use a new data root: {saved}")
        validate_inventory(selected)
        readback = destination / "inventory.csv"
        write_csv(readback, selected)
        write_json(saved, membership)
        write_json(destination / "label_map.json", {
            clique: i for i, clique in enumerate(sorted({r["clique_id"] for r in selected if r["partition"] == "train"}))
        })
        write_csv(destination / "base_train.csv", [r for r in rows if r["dataset"] == dataset and r["partition"] != "test"])
        for partition in ("train", "val", "test"):
            write_csv(destination / f"{partition}.csv", [r for r in selected if r["partition"] == partition])
    return {"datasets": sorted({r["dataset"] for r in rows}), "inventory_tracks": len(rows)}


def partition_path(config, partition, dataset=None):
    if config["paths"]["manifest"]:
        return Path(config["paths"]["manifest"]) / f"{partition}.csv"
    return Path(config["paths"]["data"]) / "manifests" / (dataset or config["dataset"]) / f"{partition}.csv"


def load_partition(config, partition, dataset=None, modality=None, check_features=True):
    path = partition_path(config, partition, dataset)
    rows = read_inventory(path)
    selected = [r for r in rows if r["dataset"] == (dataset or config["dataset"]) and r["modality"] == (modality or config["modality"])]
    if not selected:
        raise ValueError(f"Missing required {partition} set: {dataset or config['dataset']}.{modality or config['modality']}")
    for row in selected:
        if row["partition"] != partition:
            raise ValueError(f"Incorrect partition in {path}: {row['track_id']}")
        if check_features:
            feature = Path(config["paths"]["data"]) / row.get("feature_path", "")
            if not row.get("feature_sha256") or not row.get("feature_recipe"):
                raise ValueError(f"Missing feature provenance for {row['track_id']}")
            if not feature.is_file() or sha256(feature) != row["feature_sha256"]:
                raise ValueError(f"Missing/corrupt feature: {feature}")
            array = np.load(feature, allow_pickle=False)
            if array.ndim != 2 or array.shape[0] != 84 or array.shape[1] == 0 or array.dtype != np.float64 or not np.isfinite(array).all():
                raise ValueError(f"Invalid feature array: {feature}")
            sidecar = json.loads(feature.with_suffix(".json").read_text())
            if sidecar["sha256"] != row["feature_sha256"] or sidecar["recipe_hash"] != row["feature_recipe"]:
                raise ValueError(f"Feature provenance mismatch: {feature}")
            if sidecar["shape"] != list(array.shape) or sidecar["dtype"] != str(array.dtype):
                raise ValueError(f"Feature shape/dtype provenance mismatch: {feature}")
            spec = sidecar["recipe"]
            import hashlib
            fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
            if fingerprint != row["feature_recipe"]:
                raise ValueError(f"Feature recipe fingerprint mismatch: {feature}")
            synthetic = spec.get("synthetic") is True and (Path(config["paths"]["data"]) / "fixture.json").is_file()
            if not synthetic:
                audio = Path(config["paths"]["data"]) / row["audio_path"]
                if spec.get("audio_sha256") != sha256(audio):
                    raise ValueError(f"Audio provenance mismatch: {audio}")
    return selected


def validate_partitions(config):
    sets = [(config["dataset"], config["modality"], part) for part in ("train", "val")]
    sets += [(*name.split("."), "test") for name in config["evaluation"]["sets"]]
    partitions, counts, hashes = {}, {}, {}
    for dataset, modality, part in sets:
        rows = load_partition(config, part, dataset, modality)
        counts[f"{dataset}.{modality}.{part}"] = len(rows)
        hashes[str(partition_path(config, part, dataset))] = sha256(partition_path(config, part, dataset))
        existing = partitions.setdefault(dataset, {})
        existing.setdefault(part, []).extend(rows)
    for dataset, parts in partitions.items():
        clique_sets = {p: {r["clique_id"] for r in rows} for p, rows in parts.items()}
        family_sets = {p: {r["source_track_id"] for r in rows} for p, rows in parts.items()}
        for a in parts:
            for b in parts:
                if a != b and (clique_sets[a] & clique_sets[b] or family_sets[a] & family_sets[b]):
                    raise ValueError(f"Partition leakage: {dataset} {a}/{b}")
        # Check the complete saved inventory, not only the selected modalities.
        directory = partition_path(config, "test", dataset).parent
        identity_meanings = {}
        all_rows = []
        for part in ("train", "val", "test"):
            path = directory / f"{part}.csv"
            if path.exists():
                for row in read_inventory(path):
                    if row["partition"] != part:
                        raise ValueError(f"Incorrect partition in {path}")
                    all_rows.append(row)
                    key = row["track_id"]
                    meaning = (row["clique_id"], row["source_track_id"], row["partition"], row["is_original"])
                    if key in identity_meanings and identity_meanings[key] != meaning:
                        raise ValueError(f"Cross-modality membership mismatch: {dataset}.{key}")
                    identity_meanings[key] = meaning
        validate_inventory(all_rows)
        for part in ("train", "val", "test"):
            per_modality = {
                modality: {r["track_id"] for r in all_rows if r["partition"] == part and r["modality"] == modality}
                for modality in ("mix", "vocal")
            }
            if all(per_modality.values()) and per_modality["mix"] != per_modality["vocal"]:
                raise ValueError(f"Cross-modality track membership differs: {dataset}.{part}")
        saved_split = directory / "split.json"
        if not saved_split.is_file():
            raise ValueError(f"Missing persisted split provenance: {saved_split}")
        split = json.loads(saved_split.read_text())
        actual_test = sorted({r["track_id"] for r in read_inventory(directory / "test.csv")})
        if actual_test != split["test_tracks"]:
            raise ValueError(f"Test membership differs from preserved inventory: {dataset}")
        for part in ("train", "val", "test"):
            actual = sorted(f"{r['modality']}.{r['track_id']}" for r in read_inventory(directory / f"{part}.csv"))
            if actual != split["memberships"][part]:
                raise ValueError(f"Persisted {part} membership differs: {dataset}")
        hashes[str(saved_split)] = sha256(saved_split)
        label_map = directory / "label_map.json"
        expected_labels = {
            label: i for i, label in enumerate(sorted({r["clique_id"] for r in all_rows if r["partition"] == "train"}))
        }
        if json.loads(label_map.read_text()) != expected_labels:
            raise ValueError(f"Persisted label map differs: {dataset}")
        hashes[str(label_map)] = sha256(label_map)
    return {"counts": counts, "manifest_hashes": hashes}
