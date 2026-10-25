import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from csi import checkpoints
from csi.artifacts import sha256, verify
from csi.config import resolve
from csi.data.manifests import FEATURES, REQUIRED, prepare, read_inventory, write_csv
from csi.data.fixtures import create_fixture
from csi.models.adapters import ExperimentModel
from csi.training import preflight, train


def fixture_data(tmp_path):
    root = tmp_path / "data"
    create_fixture(root)
    inventory = root / "metadata/tracks.csv"
    return root, inventory


def tiny_config(tmp_path, preset="prerequisites.ce"):
    root, _ = fixture_data(tmp_path)
    return resolve(preset, root=tmp_path, overrides=[
        f"paths.data={root}", "train.batch_size=4", "train.epochs=2",
        "preprocess.frames=200", "train.workers=0",
        "model.replace_classifier=true",
    ])


def test_manifest_split_reuse_and_leak_detection(tmp_path):
    root, inventory = fixture_data(tmp_path)
    split = {"validation_fraction": 0.1, "seed": 42, "rounding": "ceil"}
    prepare(inventory, root, split)
    before = sha256(root / "manifests/indian/train.csv")
    prepare(inventory, root, split)
    assert sha256(root / "manifests/indian/train.csv") == before
    with pytest.raises(ValueError, match="differs"):
        prepare(inventory, root, {**split, "seed": 99})
    rows = read_inventory(inventory)
    rows[0]["partition"] = "test"
    write_csv(inventory, rows)
    with pytest.raises(ValueError, match="crosses|mismatch"):
        read_inventory(inventory)


def test_checkpoint_strictness():
    model = ExperimentModel(2, resolve("R1")["config"]["model"])
    state = model.backbone.state_dict()
    other = {**state, "unexpected": torch.zeros(1)}
    with pytest.raises(ValueError, match="keys differ"):
        checkpoints.warm_start(model, other)
    other = dict(state)
    other["fc0.bias"] = torch.zeros(2)
    with pytest.raises(ValueError, match="Incompatible"):
        checkpoints.warm_start(model, other, True)
    other = dict(state)
    other["fc1.bias"] = torch.zeros(4)
    other["fc1.weight"] = torch.zeros(4, 300)
    with pytest.raises(ValueError, match="Incompatible"):
        checkpoints.warm_start(model, other)
    events = checkpoints.warm_start(model, other, True)
    assert len(events) == 2


def test_cpu_training_resume_exact_and_artifact_integrity(tmp_path):
    torch.set_num_threads(2)
    resolved = tiny_config(tmp_path)
    complete = train(resolved, "complete")
    first = train(resolved, "interrupted", stop_after=1)
    continued = train(resolved, "continued", resume=Path(first["run"]) / "checkpoints/last.pt")
    expected = checkpoints.load(Path(complete["run"]) / "checkpoints/last.pt")
    actual = checkpoints.load(Path(continued["run"]) / "checkpoints/last.pt")
    for key in expected["model"]:
        torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
    assert actual["history"] == expected["history"]
    assert actual["scheduler"] == expected["scheduler"]
    assert actual["stale"] == expected["stale"]
    assert set(continued["metrics"]) == {"indian.mix", "indian.vocal", "shs100k.mix", "shs100k.vocal"}
    assert verify(continued["run"])["verified"]
    metrics = Path(continued["run"]) / "metrics.json"
    metrics.write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        verify(continued["run"])


def test_dynamic_modified_off_matrix_smoke(tmp_path):
    torch.set_num_threads(2)
    resolved = tiny_config(tmp_path, "prerequisites.modified_dynamic")
    config = resolved["config"]
    config["train"]["epochs"] = 1
    config["sampler"].update(kind="balanced", cliques_per_batch=2, tracks_per_clique=2)
    config["objectives"]["modified_triplet"]["positive"] = "hard_mean"
    config["objectives"]["supcon"]["weight"] = 0.15
    config["objectives"]["arcface"]["weight"] = 0.5
    result = train(resolved, "custom")
    state = checkpoints.load(Path(result["run"]) / "checkpoints/last.pt")
    assert state["negative_cache"]["selection_epoch"] == 0
    batch = state["history"][0]["batches"][0]
    assert batch["valid_objectives"] == 3
    assert all(batch[k]["valid"] > 0 for k in ("modified_triplet", "supcon", "arcface"))


def assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_nested_equal(a, e)
    else:
        assert actual == expected


def test_resume_heads_sampler_dropout_and_refresh_boundary(tmp_path):
    torch.set_num_threads(2)
    resolved = tiny_config(tmp_path, "prerequisites.modified_dynamic")
    config = resolved["config"]
    config["train"]["epochs"] = 4
    config["model"]["dropout"] = 0.3
    config["sampler"].update(kind="balanced", cliques_per_batch=2, tracks_per_clique=2)
    config["objectives"]["modified_triplet"]["positive"] = "hard_mean"
    config["objectives"]["supcon"]["weight"] = 0.15
    config["objectives"]["arcface"]["weight"] = 0.5
    full = train(resolved, "full")
    partial = train(resolved, "partial", stop_after=3)
    resumed = train(resolved, "resumed", resume=Path(partial["run"]) / "checkpoints/last.pt")
    expected = checkpoints.load(Path(full["run"]) / "checkpoints/last.pt")
    actual = checkpoints.load(Path(resumed["run"]) / "checkpoints/last.pt")
    for key in ("model", "objectives", "optimizer", "scheduler", "rng", "sampler", "negative_cache", "stale", "history"):
        assert_nested_equal(actual[key], expected[key])
    assert actual["history"][3]["refresh"] == ["negative_identities", "negative_embeddings"]
    with pytest.raises(ValueError, match="evaluation snapshot"):
        train(resolved, "invalid-resume", resume=Path(full["run"]) / "checkpoints/best.pt")


@pytest.mark.parametrize("preset", ["prerequisites.triplet", "prerequisites.modified_static"])
def test_prerequisite_cpu_smoke(tmp_path, preset):
    torch.set_num_threads(2)
    resolved = tiny_config(tmp_path, preset)
    resolved["config"]["train"]["epochs"] = 1
    assert train(resolved, "prerequisite")["completed_epochs"] == 1


def test_doctor_fails_for_missing_set_before_run(tmp_path):
    resolved = tiny_config(tmp_path)
    path = Path(resolved["config"]["paths"]["data"]) / "manifests/shs100k/test.csv"
    path.unlink()
    with pytest.raises(OSError):
        preflight(resolved)
    assert not (tmp_path / "runs").exists()
