"""
Audit vocal separation outputs.

Compares input audio files vs output vocal files and reports:
  - Missing files (in input but not in output)
  - Corrupt/unreadable files (output exists but can't be loaded)
  - Size anomalies (output is suspiciously small vs input)

Usage:
  python scripts/vocalseparation/audit_vocal_sep.py
  python scripts/vocalseparation/audit_vocal_sep.py --fix   # re-queues missing for re-run
"""

import argparse
from pathlib import Path
import torchaudio
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_CONFIG = {
    "indian_test": {
        "input_dir":  PROJECT_ROOT / "data" / "test" / "indian",
        "output_dir": PROJECT_ROOT / "data" / "test" / "indian_vocal",
    },
    "shs100k_test": {
        "input_dir":  PROJECT_ROOT / "data" / "test" / "shs100k-test",
        "output_dir": PROJECT_ROOT / "data" / "test" / "shs100k_test_vocal",
    },
    "shs100k_train": {
        "input_dir":  PROJECT_ROOT / "data" / "train" / "shs100k_train",
        "output_dir": PROJECT_ROOT / "data" / "train" / "shs100k_train_vocal",
    },
}

MIN_SIZE_BYTES = 10_000   # files under 10 KB are considered suspect
SIZE_RATIO_THRESHOLD = 0.01  # output < 1% of input size is suspicious


def check_file(out_path: Path, in_path: Path):
    """Returns (status, message) where status is 'ok', 'corrupt', or 'small'."""
    out_size = out_path.stat().st_size
    in_size  = in_path.stat().st_size if in_path.exists() else None

    if out_size < MIN_SIZE_BYTES:
        return "small", f"Only {out_size} bytes"

    if in_size and in_size > 0 and (out_size / in_size) < SIZE_RATIO_THRESHOLD:
        return "small", f"Output {out_size}B vs input {in_size}B ({out_size/in_size*100:.2f}%)"

    try:
        info = torchaudio.info(str(out_path))
        if info.num_frames == 0:
            return "corrupt", "Zero audio frames"
        return "ok", f"{info.num_frames/info.sample_rate:.1f}s"
    except Exception as e:
        return "corrupt", str(e)


def audit_dataset(name: str, config: dict, workers: int = 8):
    input_dir  = config["input_dir"]
    output_dir = config["output_dir"]

    print(f"\n{'='*60}")
    print(f"Auditing: {name}")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")

    if not input_dir.exists():
        print(f"  ERROR: Input directory not found!")
        return []

    extensions = ["*.mp3", "*.wav", "*.flac", "*.ogg"]
    input_files = []
    for ext in extensions:
        input_files.extend(input_dir.glob(ext))
    input_files.sort()

    issues = []

    # Step 1: Find missing files
    missing = []
    for f in input_files:
        out = output_dir / f.name
        if not out.exists():
            missing.append(f.name)

    if missing:
        issues.extend([("missing", name, f) for f in missing])
        print(f"  MISSING ({len(missing)}):")
        for f in missing:
            print(f"    - {f}")
    else:
        print(f"  Missing: 0 ✓")

    # Step 2: Check existing output files for corruption/size issues
    output_files = list(output_dir.glob("*.mp3")) + list(output_dir.glob("*.wav"))
    output_files.sort()

    if not output_files:
        print(f"  No output files found in {output_dir}")
        return issues

    print(f"  Checking {len(output_files)} output files for corruption...")

    def check_task(out_path):
        in_path = input_dir / out_path.name
        return out_path, check_file(out_path, in_path)

    corrupt = []
    small = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(check_task, f): f for f in output_files}
        for future in as_completed(futures):
            out_path, (status, msg) = future.result()
            if status == "corrupt":
                corrupt.append((out_path.name, msg))
                issues.append(("corrupt", name, out_path.name))
            elif status == "small":
                small.append((out_path.name, msg))
                issues.append(("small", name, out_path.name))

    if corrupt:
        print(f"  CORRUPT ({len(corrupt)}):")
        for f, msg in sorted(corrupt):
            print(f"    - {f}: {msg}")
    else:
        print(f"  Corrupt: 0 ✓")

    if small:
        print(f"  SUSPICIOUSLY SMALL ({len(small)}):")
        for f, msg in sorted(small):
            print(f"    - {f}: {msg}")
    else:
        print(f"  Small: 0 ✓")

    total_issues = len(missing) + len(corrupt) + len(small)
    print(f"  Summary: {len(input_files)} input | {len(output_files)} output | {total_issues} issues")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Audit vocal separation outputs")
    parser.add_argument("--fix", action="store_true",
                        help="Delete corrupt/small files so they get re-processed")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers for file checking (default: 8)")
    args = parser.parse_args()

    all_issues = []
    for name, config in DATASET_CONFIG.items():
        issues = audit_dataset(name, config, args.workers)
        all_issues.extend(issues)

    print(f"\n{'='*60}")
    print(f"TOTAL ISSUES: {len(all_issues)}")

    if all_issues:
        # Write a report file
        report_path = PROJECT_ROOT / "scripts" / "vocalseparation" / "missing_files.txt"
        with open(report_path, "w") as f:
            f.write("# Vocal Separation Audit Report\n")
            f.write(f"# Total issues: {len(all_issues)}\n\n")
            for issue_type, dataset, filename in sorted(all_issues):
                f.write(f"{issue_type}\t{dataset}\t{filename}\n")
        print(f"Report written to: {report_path}")

        if args.fix:
            print("\nFixing: removing corrupt/small files for re-processing...")
            fixed = 0
            for issue_type, dataset, filename in all_issues:
                if issue_type in ("corrupt", "small"):
                    out_path = DATASET_CONFIG[dataset]["output_dir"] / filename
                    if out_path.exists():
                        out_path.unlink()
                        print(f"  Deleted: {out_path}")
                        fixed += 1
            print(f"Deleted {fixed} files. Re-run separate_vocals.py to re-process them.")
    else:
        print("All datasets fully processed with no issues!")


if __name__ == "__main__":
    main()
