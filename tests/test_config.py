import json
import subprocess
import sys

import pytest

from csi.cli import main
from csi.config import registry, resolve


def test_every_preset_and_alias():
    rows = registry()
    assert len(rows) == 28
    for name, row in rows.items():
        assert resolve(name)["preset"] == name
        for alias in row.get("aliases", []):
            assert resolve(alias)["config"] == resolve(name)["config"]
    assert resolve("E4")["config"]["train"]["epochs"] == 20
    assert resolve("E5")["config"]["train"]["epochs"] == 20
    assert resolve("E4_75")["dependency"] == "phase1.r9"


@pytest.mark.parametrize("override", [
    "train.typo=1", "train.epochs=true", "train.lr=-1", "model.dropout=1",
    "negative.embedding_refresh_epochs=0", "objectives.triplet.distance=invalid",
    "sampler.kind=balanced", "objectives.supcon.temperature=0",
])
def test_invalid_overrides(override):
    # Balanced by itself is valid with the default 16x4 structure.
    if override == "sampler.kind=balanced":
        override = "sampler.cliques_per_batch=0"
    with pytest.raises(ValueError):
        resolve("R9", overrides=[override])


def test_off_matrix_and_never_refresh():
    result = resolve("modified_dynamic", overrides=[
        "objectives.modified_triplet.positive=hard_mean", "sampler.kind=balanced",
        "objectives.supcon.weight=0.15", "objectives.arcface.weight=0.5",
        "preprocess.tonic.enabled=true", "negative.selection_refresh_epochs=null",
    ])
    assert result["derived"]
    assert result["config"]["negative"]["selection_refresh_epochs"] is None


def test_planning_is_side_effect_free(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["suite", "--group", "phase3", "--plan"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 5
    assert not list(tmp_path.iterdir())


def test_config_paths(tmp_path):
    config = tmp_path / "custom.json"
    config.write_text('{"paths": {"data": "assets"}}')
    assert resolve(config_path=config)["config"]["paths"]["data"] == str(tmp_path / "assets")


def test_planning_does_not_import_compute():
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; from csi.cli import main; main(['suite', '--group', 'phase3', '--plan']); assert 'torch' not in sys.modules",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
