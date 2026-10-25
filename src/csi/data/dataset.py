import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..artifacts import sha256
from .augmentations import preprocess
from .tonic import ESTIMATOR


def tonic_key(row):
    return f"{row['dataset']}.{row['modality']}.{row['track_id']}"


class CQTTracks(Dataset):
    def __init__(self, rows, config, training=False, frames=None, labels=None):
        self.rows, self.config = rows, config
        self.training, self.frames = training, frames
        mapping = labels if labels is not None else {label: i for i, label in enumerate(sorted({r["clique_id"] for r in rows}))}
        self.labels = [mapping[r["clique_id"]] for r in rows]
        self.originals = [r["is_original"] == "true" for r in rows]
        self.tonics = None
        if config["preprocess"]["tonic"]["enabled"]:
            path = config["paths"]["tonic"]
            if not path:
                raise ValueError("Tonic is enabled; set paths.tonic to a built index")
            self.tonics = json.loads(Path(path).read_text())
            for row in rows:
                cached = self.tonics.get(tonic_key(row))
                if cached is None or cached["feature_sha256"] != row["feature_sha256"] or cached["feature_recipe"] != row["feature_recipe"]:
                    raise ValueError(f"Missing/stale tonic cache: {tonic_key(row)}")
                if type(cached["tonic"]) is not int or not 0 <= cached["tonic"] < 12:
                    raise ValueError(f"Invalid cached tonic: {tonic_key(row)}")
                if cached.get("method") != ESTIMATOR:
                    raise ValueError(f"Stale tonic estimation method: {tonic_key(row)}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(self.config["paths"]["data"]) / row["feature_path"]
        cqt = np.load(path, allow_pickle=False)
        tonic = self.tonics[tonic_key(row)]["tonic"] if self.tonics is not None else None
        data = preprocess(cqt, self.config["preprocess"], self.config["profile"], self.training, self.frames, tonic)
        return {
            "cqt": torch.from_numpy(data), "label": self.labels[index],
            "original": self.originals[index], "index": index,
        }
