"""Validate the CSV layout, finite numeric values, and optional split overlap."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_support import default_dataset_root


def inspect_dataset(root, check_overlap=False):
    result = {"splits": {}, "native_lengths": {}, "duplicate_values_across_splits": 0, "errors": []}
    seen = {}
    for split in ("train", "val", "test"):
        result["splits"][split] = {}
        for name in ("F0", "F1", "F2", "F3", "F4"):
            folder = root / split / name
            paths = sorted(folder.glob("*.csv"))
            result["splits"][split][name] = len(paths)
            if not paths:
                result["errors"].append(f"No CSV samples in {split}/{name}")
            for path in paths:
                try:
                    with path.open(encoding="utf-8-sig", newline="") as handle:
                        values = [float(item) for row in csv.reader(handle) for item in row if item.strip()]
                    if not values or not all(math.isfinite(x) for x in values):
                        raise ValueError("empty or non-finite sample")
                    length = str(len(values))
                    result["native_lengths"][length] = result["native_lengths"].get(length, 0) + 1
                    if check_overlap:
                        digest = hashlib.sha256(repr(values).encode()).hexdigest()
                        if digest in seen and split not in seen[digest]:
                            result["duplicate_values_across_splits"] += 1
                        seen.setdefault(digest, set()).add(split)
                except (OSError, ValueError) as error:
                    result["errors"].append(f"{path.relative_to(root).as_posix()}: {error}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--check-overlap", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect_dataset(args.dataset_root, args.check_overlap)
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    if result["errors"]:
        raise SystemExit(1)
    if result["duplicate_values_across_splits"]:
        print("Exact repeated signal values occur across splits; inspect acquisition-level independence.")


if __name__ == "__main__":
    main()
