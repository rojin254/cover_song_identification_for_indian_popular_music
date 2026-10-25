import random
import pickle
from pathlib import Path

import numpy as np
import torch


def rng_state():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    name, values, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG state cannot be restored on a CPU-only host")
        torch.cuda.set_rng_state_all(state["cuda"])


def save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as exc:
        raise ValueError(f"Invalid or unsupported tensor checkpoint: {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint must contain a state dictionary: {path}")
    return state


def require_resume_state(state):
    if state.get("kind") == "evaluation":
        raise ValueError("best.pt is an evaluation snapshot; --resume requires the full last.pt state")
    required = {
        "config", "model", "objectives", "optimizer", "scheduler", "rng", "sampler", "negative_cache",
        "next_epoch", "best", "threshold_best", "stale", "best_epoch", "best_model", "best_objectives",
        "history", "input_hashes", "source_hashes", "label_map",
    }
    if state.get("format") != 1 or not required.issubset(state):
        raise ValueError("Resume requires a toolkit last.pt containing the full training state; use --init for weights only")


def warm_start(model, checkpoint, replace_classifier=False):
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise ValueError("Expected a tensor state dictionary")
    normalized, events = {}, []
    for key, tensor in state.items():
        normalized_key = key.removeprefix("module.").removeprefix("backbone.")
        if normalized_key in normalized:
            raise ValueError(f"Checkpoint prefix collision: {key}")
        normalized[normalized_key] = tensor
        if normalized_key != key:
            events.append(f"{key} -> {normalized_key}")
    target = model.backbone.state_dict()
    if set(normalized) != set(target):
        raise ValueError(f"Checkpoint keys differ: missing={set(target)-set(normalized)}, unexpected={set(normalized)-set(target)}")
    for key, value in normalized.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Checkpoint entry is not a tensor: {key}")
        if replace_classifier and key in ("fc1.weight", "fc1.bias"):
            normalized[key] = target[key]
            events.append(f"Explicit classifier replacement: {key}")
        elif value.shape != target[key].shape:
            raise ValueError(f"Incompatible checkpoint tensor: {key}: {value.shape} vs {target[key].shape}")
    model.backbone.load_state_dict(normalized, strict=True)
    return events
