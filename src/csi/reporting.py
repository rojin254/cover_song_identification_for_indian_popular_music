import json
import copy
import hashlib
from importlib.resources import files
from pathlib import Path
from statistics import mean, stdev

from .artifacts import verify, write_json


def summarize(root, output=None):
    targets = json.loads(files("csi").joinpath("presets/report_targets.json").read_text())
    rows, groups = [], {}
    for config_file in sorted(Path(root).glob("*/config.json")):
        run = config_file.parent
        if not (run / "metrics.json").is_file():
            continue
        verify(run, check_inputs=False)
        resolved = json.loads(config_file.read_text())
        metrics = json.loads((run / "metrics.json").read_text())
        history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()] if (run / "history.jsonl").is_file() else []
        preset = resolved["preset"]
        target = targets["principal"].get(preset, {})
        row = {
            "run": str(run), "preset": preset, "derived": resolved["derived"],
            "metrics": metrics, "epochs": len(history),
            "best_validation_MAP": max((r["validation_MAP"] for r in history), default=None),
            "report_delta": {name: values["MAP"] - target[name] for name, values in metrics.items() if name in target},
            "comparison_is_reproduction": False,
        }
        rows.append(row)
        config = resolved["config"]
        if history:
            comparable = copy.deepcopy(config)
            comparable.pop("paths")
            comparable["train"].pop("seed")
            signature = hashlib.sha256(json.dumps(comparable, sort_keys=True).encode()).hexdigest()[:16]
            key = f"{preset}.{config['dataset']}.{config['modality']}.{signature}"
            groups.setdefault(key, []).append(row)
    result = {
        "runs": rows,
        "statistics": {
            key: {name: {
                "n": len(group),
                "MAP_mean": mean(row["metrics"][name]["MAP"] for row in group),
                "MAP_sample_std": stdev(row["metrics"][name]["MAP"] for row in group) if len(group) > 1 else None,
            } for name in group[0]["metrics"] if all(name in row["metrics"] for row in group)}
            for key, group in groups.items()
        },
    }
    if output:
        write_json(output, result)
    return result


def reevaluate(run, output_run_id, atol):
    import math
    from .training import evaluate
    if not output_run_id or not math.isfinite(atol) or atol < 0:
        raise ValueError("Reevaluation requires --output-run-id and finite nonnegative --atol")
    run = Path(run)
    resolved = json.loads((run / "config.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    checkpoint = run / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        external = provenance.get("checkpoint")
        if external is None:
            raise ValueError("Run has no best checkpoint or external evaluation checkpoint")
        checkpoint = Path(external["path"])
        from .artifacts import sha256
        if sha256(checkpoint) != external["sha256"]:
            raise ValueError("External evaluation checkpoint has changed")
    result = evaluate(resolved, checkpoint, output_run_id)
    previous = json.loads((run / "metrics.json").read_text())
    for name, metrics in previous.items():
        for metric in ("MAP", "Top10", "Rank1"):
            if abs(metrics[metric] - result["metrics"][name][metric]) > atol:
                raise ValueError(f"Reevaluation mismatch: {name}.{metric}; output retained at {result['run']}")
    result["within_tolerance"] = True
    return result
