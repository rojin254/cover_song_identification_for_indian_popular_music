import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from csi.cli import main, parser
from csi.config import resolve
from csi.data.fixtures import create_fixture
from csi.data.operations import build_tonic, prepare_data
from csi.reporting import summarize
from csi.training import evaluate, train


def test_cli_fixture_prepare_doctor_tonic(tmp_path, capsys):
    root = tmp_path / "fixture"
    assert main(["data", "fixture", "--output", str(root)]) == 0
    common = ["--root", str(tmp_path), "--set", f"paths.data={root}", "--set", "train.batch_size=4"]
    assert main(["data", "prepare", "--inventory", str(root / "metadata/tracks.csv"), *common]) == 0
    tonic = root / "cache/tonic/all.json"
    assert main(["data", "tonic", "--output", str(tonic), *common]) == 0
    assert main(["data", "tonic", "--output", str(tonic), "--check", *common]) == 0
    assert main(["doctor", *common, "--set", "preprocess.tonic.enabled=true", "--set", f"paths.tonic={tonic}"]) == 0
    data = json.loads(tonic.read_text())
    data.pop(next(iter(data)))
    tonic.write_text(json.dumps(data))
    assert main(["doctor", *common, "--set", "preprocess.tonic.enabled=true", "--set", f"paths.tonic={tonic}"]) == 2
    assert "tonic" in capsys.readouterr().err


def test_supcon_only_cli_train_evaluate_verify_summary(tmp_path, capsys):
    import torch
    torch.set_num_threads(2)
    root = tmp_path / "data"
    create_fixture(root)
    common = [
        "--root", str(tmp_path), "--preset", "prerequisites.ce", "--set", "train.batch_size=4",
        "--set", "preprocess.frames=200", "--set", "train.epochs=1",
        "--set", "objectives.ce.weight=0", "--set", "objectives.supcon.weight=1",
    ]
    assert main(["train", "--run-id", "supcon", *common]) == 0
    checkpoint = tmp_path / "runs/supcon/checkpoints/best.pt"
    assert main(["evaluate", "--run-id", "evaluation", "--init", str(checkpoint), *common]) == 0
    run = tmp_path / "runs/supcon"
    assert main(["verify", "--run", str(run), "--reevaluate", "--output-run-id", "parity"]) == 0
    output = tmp_path / "summary.json"
    assert main(["summarize", "--runs", str(tmp_path / "runs"), "--output", str(output)]) == 0
    assert len(json.loads(output.read_text())["runs"]) == 3
    capsys.readouterr()


def test_upstream_three_length_ce_smoke(tmp_path):
    import torch
    torch.set_num_threads(2)
    create_fixture(tmp_path / "data")
    resolved = resolve("upstream.ce", root=tmp_path, overrides=["train.batch_size=4", "train.epochs=1"])
    result = train(resolved, "upstream")
    history = json.loads((Path(result["run"]) / "history.jsonl").read_text())
    assert len(history["batches"]) == history["updates"] == 3
    assert all("ce" in batch for batch in history["batches"])


def test_audio_feature_extraction_and_recorded_recipe(tmp_path):
    import wave
    from csi.data.cqt import extract
    from csi.data.operations import feature_job
    sr = 22050
    signal = np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr)
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sr)
        stream.writeframes((signal * 30000).astype("<i2").tobytes())
    settings = resolve()["config"]["preprocess"]
    result = feature_job(({
        "dataset": "indian", "modality": "mix", "track_id": "tone", "audio_path": "tone.wav",
    }, str(tmp_path), settings))
    array = np.load(tmp_path / result["feature_path"])
    assert array.shape[0] == 84 and array.dtype == np.float64
    assert feature_job(({"dataset": "indian", "modality": "mix", "track_id": "tone", "audio_path": "tone.wav"}, str(tmp_path), settings)) == result
    sidecar = json.loads((tmp_path / result["feature_path"]).with_suffix(".json").read_text())
    assert "librosa" in sidecar["recipe"]["versions"]
    np.testing.assert_array_equal(array, extract(path, settings))


def test_suite_requires_explicit_execution_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["suite", "--group", "phase3", "--execute"]) == 2
    assert not list(tmp_path.iterdir())


def test_installed_cli_cpu_workflow_outside_repository(tmp_path):
    console = Path(sys.executable).with_name("csi")
    assert console.is_file(), "Install the toolkit before running CLI smoke tests"
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

    def run(*arguments, module=False):
        command = [sys.executable, "-m", "csi"] if module else [str(console)]
        result = subprocess.run(
            [*command, *arguments], cwd=tmp_path, env=environment,
            capture_output=True, text=True, timeout=90,
        )
        assert result.returncode == 0, f"{arguments}\n{result.stdout}\n{result.stderr}"
        return json.loads(result.stdout)

    assert len(run("list", module=True)["presets"]) == 28
    fixture = tmp_path / "data"
    assert run("data", "fixture", "--output", str(fixture))["synthetic"]
    common = [
        "--preset", "prerequisites.ce", "--root", str(tmp_path),
        "--set", "train.device=cpu", "--set", "train.batch_size=4",
        "--set", "preprocess.frames=200", "--set", "train.epochs=1",
    ]
    run("data", "prepare", "--inventory", str(fixture / "metadata/tracks.csv"), *common)
    assert run("doctor", *common)["classes"] == 2
    trained = run("train", "--run-id", "cli-ce", *common)
    assert trained["completed_epochs"] == 1
    checkpoint = tmp_path / "runs/cli-ce/checkpoints"
    resumed = run("train", "--run-id", "cli-resume", "--resume", str(checkpoint / "last.pt"), *common)
    assert resumed["completed_epochs"] == 1
    assert resumed["metrics"] == trained["metrics"]
    evaluated = run("evaluate", "--run-id", "cli-eval", "--init", str(checkpoint / "best.pt"), *common)
    assert evaluated["metrics"] == trained["metrics"]
    verified = run(
        "verify", "--run", str(tmp_path / "runs/cli-ce"),
        "--reevaluate", "--output-run-id", "cli-parity",
    )
    assert verified["verified"] and verified["reevaluation"]["within_tolerance"]
    summary = run("summarize", "--runs", str(tmp_path / "runs"))
    assert len(summary["runs"]) == 4
