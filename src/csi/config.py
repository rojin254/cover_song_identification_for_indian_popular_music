"""Strict JSON configuration; planning does not import the compute runtime."""

import copy
import json
import math
from importlib.resources import files
from pathlib import Path

SOURCE_COMMIT = "4a636cf0944e7c6b60f838845b8073fafb06ee59"
UPSTREAM_COMMIT = "c6411f41258b9cd5e8ce8f000d7f494391b4f6e2"

DEFAULTS = {
    "dataset": "indian",
    "modality": "mix",
    "profile": "report",
    "paths": {"data": "data", "runs": "runs", "manifest": "", "tonic": ""},
    "preprocess": {
        "frames": 400,
        "tonic": {"enabled": False},
        "augmentation": False,
        "sample_rate": 22050,
        "hop_length": 512,
        "fmin": 32.70319566257483,
        "bins": 84,
        "bins_per_octave": 12,
        "aggregate": 20,
        "dtype": "float64",
    },
    "model": {
        "dropout": 0.0,
        "freeze_conv0": False,
        "batchnorm": "train",
        "replace_classifier": False,
    },
    "objectives": {
        "ce": {"weight": 0.0},
        "triplet": {"weight": 0.0, "margin": 0.3, "distance": "euclidean", "normalize": False},
        "modified_triplet": {"weight": 1.0, "margin": 0.3, "positive": "random"},
        "supcon": {"weight": 0.0, "temperature": 0.07},
        "arcface": {"weight": 0.0, "subcenters": 3, "margin": 0.3, "scale": 30.0},
    },
    "negative": {
        "selection_refresh_epochs": 3,
        "embedding_refresh_epochs": 3,
        "embedding_policy": "cached",
    },
    "mining": {"anchors": "originals", "duplicates": "exclude", "skip_batch_without_triplet": False},
    "sampler": {
        "kind": "random", "cliques_per_batch": 16, "tracks_per_clique": 4,
        "shortage": "error", "tail": "drop", "include_original": True,
    },
    "train": {
        "epochs": 75, "lr": 0.0005, "weight_decay": 0.0001, "batch_size": 64,
        "scheduler": "cosine", "min_lr": 0.000001, "plateau_factor": 0.8,
        "plateau_patience": 2, "patience": 15, "min_improvement": 0.002,
        "seed": 42, "device": "cpu", "workers": 0,
    },
    "split": {"validation_fraction": 0.1, "seed": 42, "rounding": "ceil"},
    "evaluation": {"no_positive": "zero", "sets": ["indian.mix", "indian.vocal", "shs100k.mix", "shs100k.vocal"]},
}

CHOICES = {
    "dataset": ("indian", "shs100k"), "modality": ("mix", "vocal"),
    "profile": ("report", "upstream"), "preprocess.dtype": ("float64",),
    "model.batchnorm": ("train", "frozen"),
    "objectives.triplet.distance": ("euclidean", "cosine"),
    "objectives.modified_triplet.positive": ("random", "hard", "mean", "hard_mean"),
    "negative.embedding_policy": ("cached", "live"),
    "mining.anchors": ("originals", "all"), "mining.duplicates": ("exclude",),
    "sampler.kind": ("random", "balanced"), "sampler.shortage": ("error", "replace"),
    "sampler.tail": ("drop", "error"), "train.scheduler": ("cosine", "plateau"),
    "split.rounding": ("ceil",), "evaluation.no_positive": ("zero", "error"),
}


def merge(target, update, prefix=""):
    if not isinstance(update, dict):
        raise ValueError("Configuration must be a JSON object")
    for key, value in update.items():
        field = f"{prefix}.{key}" if prefix else key
        if key not in target:
            raise ValueError(f"Unknown configuration field: {field}")
        expected = target[key]
        if isinstance(expected, dict):
            if not isinstance(value, dict):
                raise ValueError(f"{field} must be an object")
            merge(expected, value, field)
        else:
            valid = type(value) is type(expected)
            if isinstance(expected, float):
                valid = type(value) in (int, float) and math.isfinite(value)
            if field in ("negative.selection_refresh_epochs", "negative.embedding_refresh_epochs"):
                valid = value is None or type(value) is int
            if not valid:
                raise ValueError(f"Invalid type for {field}: {value!r}")
            if isinstance(expected, list) and not all(isinstance(v, str) for v in value):
                raise ValueError(f"{field} requires strings")
            target[key] = value


def flatten(config, prefix=""):
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from flatten(value, path)
        else:
            yield path, value


def validate(config):
    for field, value in flatten(config):
        if field in CHOICES and value not in CHOICES[field]:
            raise ValueError(f"{field} must be one of {CHOICES[field]}")
        if type(value) in (int, float) and value < 0:
            raise ValueError(f"{field} must be nonnegative")
        if field.endswith("_refresh_epochs") and value == 0:
            raise ValueError(f"{field}: use null for never-refresh, not zero")
    positive = (
        "preprocess.frames", "preprocess.sample_rate", "preprocess.hop_length", "preprocess.fmin",
        "preprocess.bins", "preprocess.bins_per_octave", "preprocess.aggregate",
        "train.lr", "train.batch_size", "sampler.cliques_per_batch", "sampler.tracks_per_clique",
        "objectives.supcon.temperature", "objectives.arcface.subcenters", "objectives.arcface.scale",
        "objectives.triplet.margin",
    )
    flat = dict(flatten(config))
    for field in positive:
        if flat[field] <= 0:
            raise ValueError(f"{field} must be positive")
    for field in ("model.dropout", "split.validation_fraction", "train.plateau_factor"):
        if not 0 <= flat[field] < 1:
            raise ValueError(f"{field} must be in [0, 1)")
    if config["preprocess"]["bins"] != 84 or config["preprocess"]["bins_per_octave"] != 12:
        raise ValueError("CQTNet requires 84 bins and 12 bins per octave")
    if config["preprocess"]["frames"] < 200:
        raise ValueError("CQTNet requires at least 200 frames")
    device = config["train"]["device"]
    if device != "cpu" and device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise ValueError("train.device must be cpu, cuda or cuda:<index>")
    if config["train"]["min_lr"] > config["train"]["lr"]:
        raise ValueError("train.min_lr cannot exceed train.lr")
    if config["objectives"]["arcface"]["margin"] >= math.pi:
        raise ValueError("ArcFace margin must be less than pi")
    if config["train"]["epochs"] and not any(v["weight"] for v in config["objectives"].values()):
        raise ValueError("Training requires at least one enabled objective")
    if config["sampler"]["kind"] == "balanced":
        size = config["sampler"]["cliques_per_batch"] * config["sampler"]["tracks_per_clique"]
        if size != config["train"]["batch_size"]:
            raise ValueError("Balanced batch structure must equal train.batch_size")
    if config["profile"] == "upstream":
        if config["sampler"]["kind"] != "random" or config["preprocess"]["tonic"]["enabled"]:
            raise ValueError("Upstream profile requires random sampling without tonic")
        if config["objectives"]["ce"]["weight"] != 1 or any(
            v["weight"] for k, v in config["objectives"].items() if k != "ce"
        ):
            raise ValueError("Upstream profile requires CE only")
    if not config["evaluation"]["sets"] or any(
        v not in ("indian.mix", "indian.vocal", "shs100k.mix", "shs100k.vocal")
        for v in config["evaluation"]["sets"]
    ):
        raise ValueError("evaluation.sets requires canonical dataset.modality names")
    return config


def registry():
    rows = {}
    for group in ("upstream", "prerequisites", "phase1", "phase2", "phase3"):
        rows.update(json.loads(files("csi").joinpath(f"presets/{group}.json").read_text()))
    return rows


def resolve(preset=None, config_path=None, overrides=(), root=None):
    rows = registry()
    alias = {a.lower(): name for name, row in rows.items() for a in row.get("aliases", [])}
    name = alias.get((preset or "").lower(), preset)
    config = copy.deepcopy(DEFAULTS)
    evidence = []

    def inherit(key, seen=()):
        if key not in rows or key in seen:
            raise ValueError(f"Unknown or circular preset: {key}")
        row = rows[key]
        if row.get("extends"):
            inherit(row["extends"], (*seen, key))
        merge(config, row.get("config", {}))
        evidence.extend(row.get("evidence", []))

    if name:
        inherit(name)
    baseline = copy.deepcopy(config)
    if config_path:
        path = Path(config_path).resolve()
        merge(config, json.loads(path.read_text()))
        root = root or path.parent
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep:
            raise ValueError(f"Expected field=JSON-value: {override}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        patch = value
        for part in reversed(key.split(".")):
            patch = {part: patch}
        merge(config, patch)
    validate(config)
    project_root = Path(root or Path.cwd()).resolve()
    differences = {k: v for k, v in flatten(config) if dict(flatten(baseline))[k] != v}
    for key, value in config["paths"].items():
        if value:
            path = Path(value)
            config["paths"][key] = str(path if path.is_absolute() else project_root / path)
    return {
        "preset": name, "derived": bool(differences), "differences": differences,
        "config": config, "root": str(project_root),
        "dependency": rows.get(name, {}).get("dependency"),
        "evidence": list(dict.fromkeys(evidence)),
        "source_commit": SOURCE_COMMIT, "upstream_commit": UPSTREAM_COMMIT,
        "historical_reproduction": False,
    }
