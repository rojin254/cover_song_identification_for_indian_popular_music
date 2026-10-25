"""Command-line entry point. Compute imports are deliberately lazy."""

import argparse
import json
import sys

from .config import CHOICES, DEFAULTS, flatten, registry, resolve


def parser():
    root = argparse.ArgumentParser(prog="csi", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("schema")
    def configuration(command):
        command.add_argument("--preset")
        command.add_argument("--config")
        command.add_argument("--root")
        command.add_argument("--dataset", choices=("indian", "shs100k"))
        command.add_argument("--modality", choices=("mix", "vocal"))
        command.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")

    for name in ("show", "doctor", "train", "evaluate", "suite"):
        command = commands.add_parser(name)
        configuration(command)
        if name in ("train", "evaluate", "doctor"):
            state = command.add_mutually_exclusive_group()
            state.add_argument("--init")
            state.add_argument("--resume")
        if name in ("train", "evaluate"):
            command.add_argument("--run-id", required=True)
        if name == "suite":
            command.add_argument("--group", choices=("upstream", "prerequisites", "phase1", "phase2", "phase3"))
            mode = command.add_mutually_exclusive_group(required=True)
            mode.add_argument("--plan", action="store_true")
            mode.add_argument("--execute", action="store_true")
            command.add_argument("--init-map", help="JSON mapping each selected preset to its explicit initializer path")
            command.add_argument("--run-prefix")
    data = commands.add_parser("data").add_subparsers(dest="operation", required=True)
    fixture = data.add_parser("fixture")
    fixture.add_argument("--output", required=True)
    fetch = data.add_parser("fetch")
    fetch.add_argument("--asset", choices=("dataset", "checkpoints"), required=True)
    fetch.add_argument("--file", action="append", required=True)
    fetch.add_argument("--output", required=True)
    prepare = data.add_parser("prepare")
    configuration(prepare)
    prepare.add_argument("--inventory", required=True)
    prepare.add_argument("--features", action="store_true")
    tonic = data.add_parser("tonic")
    configuration(tonic)
    tonic.add_argument("--output", required=True)
    tonic.add_argument("--check", action="store_true")
    separate = data.add_parser("separate-vocals")
    separate.add_argument("--input", action="append", required=True)
    separate.add_argument("--output", required=True)
    separate.add_argument("--model", default="htdemucs")
    separate.add_argument("--device", default="cpu")
    separate.add_argument("--allow-model-download", action="store_true")
    summary = commands.add_parser("summarize")
    summary.add_argument("--runs", default="runs")
    summary.add_argument("--output")
    verify = commands.add_parser("verify")
    verify.add_argument("--run", required=True)
    verify.add_argument("--reevaluate", action="store_true")
    verify.add_argument("--output-run-id")
    verify.add_argument("--atol", type=float, default=1e-8)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        overrides = list(getattr(args, "set", []))
        for field in ("dataset", "modality"):
            if getattr(args, field, None):
                overrides.append(f"{field}={getattr(args, field)}")
        if args.command == "list":
            result = {
                "presets": registry(), "groups": ["upstream", "prerequisites", "phase1", "phase2", "phase3"],
                "components": {
                    "objectives": list(DEFAULTS["objectives"]), "positives": CHOICES["objectives.modified_triplet.positive"],
                    "sampling": CHOICES["sampler.kind"], "tonic": "within-octave magnitude pitch-class rotation",
                },
            }
        elif args.command == "schema":
            result = {
                key: {"type": type(value).__name__, "default": value, "choices": CHOICES.get(key)}
                for key, value in flatten(DEFAULTS)
            }
        elif args.command == "suite":
            names = [n for n in registry() if not args.group or n.startswith(args.group + ".")]
            if args.preset:
                names = [resolve(args.preset)["preset"]]
            result = [resolve(n, args.config, overrides, args.root) for n in names]
            if args.execute:
                if not args.init_map or not args.run_prefix:
                    raise ValueError("Suite execution requires --init-map and --run-prefix; no initializer is guessed")
                from pathlib import Path
                from .training import preflight, train
                mapping_path = Path(args.init_map).resolve()
                initializers = json.loads(mapping_path.read_text())
                if not isinstance(initializers, dict) or any(k not in registry() or (v is not None and not isinstance(v, str)) for k, v in initializers.items()):
                    raise ValueError("--init-map must map canonical preset names to checkpoint paths or null")
                for item in result:
                    initializer = initializers.get(item["preset"])
                    if initializer is not None:
                        path = Path(initializer)
                        initializers[item["preset"]] = str(path if path.is_absolute() else mapping_path.parent / path)
                    preflight(item, init=initializers.get(item["preset"]))
                result = [train(item, f"{args.run_prefix}-{item['preset']}", init=initializers.get(item["preset"])) for item in result]
        elif args.command == "data":
            if args.operation == "fixture":
                from .data.fixtures import create_fixture
                result = create_fixture(args.output)
            elif args.operation == "fetch":
                from .data import operations
                result = operations.fetch(args.asset, args.file, args.output)
            elif args.operation == "separate-vocals":
                from .data import operations
                result = operations.separate_vocals(args.input, args.output, args.model, args.device, args.allow_model_download)
            else:
                from .data import operations
                config = resolve(args.preset, args.config, overrides, args.root)["config"]
                if args.operation == "prepare":
                    result = operations.prepare_data(args.inventory, config, args.features, config["train"]["workers"])
                else:
                    result = operations.build_tonic(config, args.output, args.check)
        elif args.command == "summarize":
            from .reporting import summarize
            result = summarize(args.runs, args.output)
        elif args.command == "verify":
            from .artifacts import verify
            result = verify(args.run)
            if args.reevaluate:
                from .reporting import reevaluate
                result["reevaluation"] = reevaluate(args.run, args.output_run_id, args.atol)
        else:
            resolved = resolve(args.preset, args.config, overrides, args.root)
            if args.command == "show":
                result = resolved
            else:
                from .training import dispatch
                result = dispatch(args, resolved)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"csi: {exc}", file=sys.stderr)
        return 2
