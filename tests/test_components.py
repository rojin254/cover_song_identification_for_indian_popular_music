import copy
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from csi.config import resolve
from csi.data.augmentations import preprocess, upstream_augment
from csi.data.cqt import aggregate_frames
from csi.data.tonic import estimate_tonic, rotate_cqt
from csi.evaluation import retrieval_metrics
from csi.losses.compose import Objectives
from csi.losses.modified_triplet import modified_triplet_loss
from csi.losses.supcon import supcon_loss
from csi.losses.triplet import triplet_loss
from csi.mining.dynamic_nsim import NegativeCache
from csi.mining.hard_mean_positive import hard_mean_positive
from csi.models import CQTNet
from csi.models.adapters import ExperimentModel
from csi.models.basic_module import BasicModule
from csi.artifacts import sha256
from csi.sampling.balanced import BalancedSampler


torch.set_num_threads(2)


def test_preserved_legacy_backbone_numeric_parity(monkeypatch):
    # Load historical source without installing its unused torchvision dependency.
    package = types.ModuleType("reference")
    package.__path__ = [str(Path(__file__).parent / "reference")]
    monkeypatch.setitem(sys.modules, "reference", package)
    monkeypatch.setitem(sys.modules, "torchvision", types.ModuleType("torchvision"))
    base = types.ModuleType("reference.basic_module")
    base.BasicModule = BasicModule
    monkeypatch.setitem(sys.modules, "reference.basic_module", base)
    spec = importlib.util.spec_from_file_location("reference.CQTNet", Path(package.__path__[0]) / "CQTNet.py")
    assert sha256(spec.origin) == "7b69cc83cb201089d3cafa3922de52f6f24f91a35dcc917b2d0eea96ba1fbf8f"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference, actual = module.CQTNet().eval(), CQTNet().eval()
    actual.load_state_dict(reference.state_dict(), strict=True)
    assert list(reference.state_dict()) == list(actual.state_dict())
    with torch.no_grad():
        for frames in (200, 300, 400, 520):
            sample = torch.randn(1, 1, 84, frames)
            for expected, result in zip(reference(sample), actual(sample)):
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
    adapter = ExperimentModel(10000, resolve("upstream")["config"]["model"]).eval()
    adapter.backbone.load_state_dict(actual.state_dict())
    with torch.no_grad():
        for expected, result in zip(actual(sample), adapter(sample)):
            torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_preprocessing_and_cqt():
    config = resolve("R1")["config"]
    cqt = np.ones((84, 405), dtype=np.float64)
    cqt[:, 404] = 10
    result = preprocess(cqt, config["preprocess"], "report")
    assert result.shape == (1, 84, 400)
    np.testing.assert_allclose(result, 0.1, atol=1e-6)
    assert preprocess(np.zeros((84, 1)), config["preprocess"], "upstream").shape == (1, 84, 200)
    raw = np.arange(84 * 43).reshape(84, 43)
    aggregated = aggregate_frames(raw)
    assert aggregated.dtype == np.float64
    np.testing.assert_array_equal(aggregated[:, 0], raw[:, :20].mean(axis=1))
    assert aggregated.shape == (84, 2)
    data = np.arange(84)[:, None]
    rotated = rotate_cqt(data, 1)
    assert rotated[11, 0] == 0 and rotated[12, 0] == 13
    tone = np.zeros((84, 2))
    tone[5::12] = 1
    assert estimate_tonic(tone) == 5


def test_upstream_masks_speed_quantization(monkeypatch):
    draws = iter((2, 1, 3, 20))
    monkeypatch.setattr(np.random, "randint", lambda upper: next(draws))
    monkeypatch.setattr(np.random, "uniform", lambda lower, upper: 0.8)
    original = np.repeat(np.arange(84, dtype=np.float64)[:, None], 250, axis=1)
    result = upstream_augment(original)
    assert result.shape == (84, 200)
    expected = original[:, 0].copy()
    expected[1:3] = 0
    expected[20:23] = 0
    expected = (expected * 255 / 84).astype(np.uint8) / 255 * 84
    np.testing.assert_allclose(result, np.repeat(expected[:, None], 200, axis=1))
    assert np.array_equal(original[:, 0], np.arange(84))


def test_hand_computed_losses_and_gradients():
    a = torch.tensor([[1., 0.]], requires_grad=True)
    p = torch.tensor([[0., 1.]], requires_grad=True)
    n = torch.tensor([[1., 0.]], requires_grad=True)
    loss = modified_triplet_loss(a, p, n, p, margin=0.3)
    torch.testing.assert_close(loss, torch.tensor(1.3))
    loss.backward()
    assert all(torch.isfinite(x.grad).all() for x in (a, p, n))
    torch.testing.assert_close(triplet_loss(a, p, n, distance="cosine"), torch.tensor(1.3))
    torch.testing.assert_close(triplet_loss(a, p, n), F.triplet_margin_loss(a, p, n, margin=0.3))
    distances = torch.tensor([[0., 0.2, 0.8]])
    mask = torch.tensor([[False, True, True]])
    positive, valid = hard_mean_positive(distances, mask)
    torch.testing.assert_close(positive, torch.tensor([1.3]))
    assert valid.all()


def test_empty_positive_and_duplicate_supcon():
    z = torch.randn(3, 300, requires_grad=True)
    loss, count = supcon_loss(z, torch.arange(3))
    assert count == 0 and loss == 0
    loss.backward()
    assert torch.isfinite(z.grad).all()
    loss, count = supcon_loss(z, torch.zeros(3, dtype=torch.long), identities=torch.zeros(3, dtype=torch.long))
    assert count == 0
    original = torch.randn(3, 300, requires_grad=True)
    labels = torch.tensor([0, 0, 1])
    expected, _ = supcon_loss(original, labels, identities=torch.arange(3))
    repeated = torch.cat((original, original[1:2]))
    actual, count = supcon_loss(repeated, torch.tensor([0, 0, 1, 0]), identities=torch.tensor([0, 1, 2, 1]))
    assert count == 2
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("objective", ["supcon", "arcface"])
def test_auxiliary_only_without_triplets(objective):
    config = resolve("R1")["config"]
    config["objectives"]["modified_triplet"]["weight"] = 0
    config["objectives"][objective]["weight"] = 1
    model = Objectives(config, 3)
    z = torch.randn(3, 300, requires_grad=True)
    labels = torch.tensor([0, 0, 2]) if objective == "supcon" else torch.arange(3)
    loss, logs = model(torch.randn(3, 3), z, labels, torch.zeros(3, dtype=torch.bool), torch.arange(3))
    assert logs["valid_objectives"] == 1 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_static_and_dynamic_refresh():
    cache = NegativeCache()
    labels = torch.tensor([0, 0, 1, 1])
    z = torch.tensor([[1., 0.], [1., .1], [1., .2], [-1., 0.]])
    static = {"selection_refresh_epochs": None, "embedding_refresh_epochs": 8}
    assert len(cache.refresh(0, z, labels, static)) == 2
    identities = cache.identities.clone()
    assert not cache.refresh(3, -z, labels, static)
    assert cache.refresh(8, -z, labels, static) == ["negative_embeddings"]
    assert torch.equal(cache.identities, identities)
    dynamic = {"selection_refresh_epochs": 3, "embedding_refresh_epochs": None}
    assert cache.refresh(9, z, labels, dynamic) == ["negative_identities"]
    assert (labels[cache.identities] != labels).all()


def test_balanced_16_by_4_repeatability_and_originals():
    labels = [i // 4 for i in range(64)]
    originals = [i % 4 == 0 for i in range(64)]
    settings = resolve("E4")["config"]["sampler"]
    sampler = BalancedSampler(labels, originals, settings)
    batch = next(iter(sampler))
    assert len(batch) == len(set(batch)) == 64
    assert all(sum(labels[i] == c for i in batch) == 4 for c in range(16))
    assert list(sampler) == list(sampler)
    state = sampler.state_dict()
    sampler.epoch = 9
    sampler.load_state_dict(state)
    assert list(sampler) == [batch]
    with pytest.raises(ValueError, match="fewer"):
        BalancedSampler(labels[:-1], originals[:-1], settings)


def test_retrieval_hand_fixture_and_no_positive():
    z = [[1, 0], [1, 0], [0, 1]]
    result = retrieval_metrics(z, ["a", "a", "b"])
    assert result["MAP"] == pytest.approx(2 / 3)
    assert result["Top10"] == pytest.approx(2 / 30)
    assert result["Rank1"] == pytest.approx(2 / 3)
    assert result["no_positive_queries"] == 1
    assert all(i not in r for i, r in enumerate(result["rankings"]))
    with pytest.raises(ValueError, match="no relevant"):
        retrieval_metrics(z, ["a", "a", "b"], "error")
