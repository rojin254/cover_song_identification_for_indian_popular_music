import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def environment():
    import torch
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    package = Path(__file__).parent
    source_hashes = {str(p.relative_to(package)): sha256(p) for p in sorted(package.rglob("*")) if p.suffix in (".py", ".json")}
    return {
        "code_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_available": revision.returncode == 0, "dirty": bool(dirty.stdout.strip()),
        "source_hashes": source_hashes,
        "versions": {name: metadata.version(name) for name in ("cqtnet-toolkit", "numpy", "torch", "pillow")},
        "installed_distributions": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu": platform.processor(), "cuda": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def seal(run):
    run = Path(run)
    hashes = {
        str(p.relative_to(run)): sha256(p) for p in sorted(run.rglob("*"))
        if p.is_file() and p.name != "artifacts.json"
    }
    write_json(run / "artifacts.json", hashes)
    return hashes


def verify(run, check_inputs=True):
    run = Path(run).resolve()
    expected = json.loads((run / "artifacts.json").read_text())
    actual = {str(p.relative_to(run)) for p in run.rglob("*") if p.is_file() and p.name != "artifacts.json"}
    if actual != set(expected):
        raise ValueError("Run artifact membership has changed")
    for name, checksum in expected.items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or sha256(path) != checksum:
            raise ValueError(f"Run artifact integrity failure: {name}")
    provenance = json.loads((run / "provenance.json").read_text())
    if not provenance.get("environment", {}).get("source_hashes"):
        raise ValueError("Missing code provenance")
    if check_inputs:
        for path, checksum in provenance["inputs"]["manifest_hashes"].items():
            if sha256(path) != checksum:
                raise ValueError(f"Input manifest integrity failure: {path}")
        from .data.manifests import load_partition, validate_partitions
        config = json.loads((run / "config.json").read_text())["config"]
        if (run / "history.jsonl").is_file():
            validate_partitions(config)
        else:
            for name in config["evaluation"]["sets"]:
                dataset, modality = name.split(".")
                load_partition(config, "test", dataset, modality)
        initializer = provenance.get("initializer") or provenance.get("checkpoint")
        if initializer and sha256(initializer["path"]) != initializer["sha256"]:
            raise ValueError("Initializer checkpoint integrity failure")
    return {"verified": True, "inputs_checked": check_inputs, "files": len(expected), "run": str(run)}
