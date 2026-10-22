"""Run isolated seeded experiments and aggregate the selected completed runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

from release_support import ROOT, default_dataset_root, source_digest


BASELINES = ("crfde", "dvpn", "gebs", "ofer", "pdagg", "sso_fc", "tnpnet", "feug", "ms_dawcae")
CASES = {"single": "F2", "double": "F2_F3", "triple": "F1_F2_F3"}
SEEDS = tuple(range(2026, 2036))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("proposed", "baselines", "all"), default="proposed")
    parser.add_argument("--models", choices=BASELINES, nargs="+", default=list(BASELINES))
    parser.add_argument("--shots", type=int, choices=(5, 10, 20), nargs="+", default=[5, 10, 20])
    parser.add_argument("--cases", choices=tuple(CASES), nargs="+", default=list(CASES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--max-discovery-per-class", type=int)
    parser.add_argument("--max-test-per-class", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("Seeds must be unique integers in [0, 2**32).")
    if args.epochs < 1 or args.num_threads < 1:
        parser.error("epochs and num-threads must be positive.")
    for key in ("max_discovery_per_class", "max_test_per_class"):
        if getattr(args, key) is not None and getattr(args, key) < 2:
            parser.error(f"{key.replace('_', '-')} must be >= 2 when supplied.")
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    return args


def make_plan(args):
    jobs = []
    families = []
    if args.method in {"proposed", "all"}:
        families.extend(("proposed", "dpc_opennet", case) for case in dict.fromkeys(args.cases))
    if args.method in {"baselines", "all"}:
        families.extend(("baselines", model, "single") for model in dict.fromkeys(args.models))
    for family, model, case in families:
        for shot in dict.fromkeys(args.shots):
            stem = CASES[case]
            script = (
                ROOT / "Experiment" / f"{shot}shot_{stem}_unknown.py"
                if family == "proposed"
                else ROOT / "compare" / f"train_{model}_{shot}shot_F2_unknown.py"
            )
            if not script.is_file():
                raise FileNotFoundError(script)
            for seed in args.seeds:
                directory = args.output_root / family / model / f"{shot}shot_{stem}_unknown" / f"seed_{seed}"
                command = [
                    sys.executable, str(script), "--dataset-root", str(args.dataset_root),
                    "--output-dir", str(directory), "--seed", str(seed),
                    "--epochs", str(args.epochs), "--device", args.device,
                    "--num-threads", str(args.num_threads),
                ]
                if family == "baselines":
                    command.append("--single-run")
                for option in ("deterministic", "skip_plots"):
                    if getattr(args, option):
                        command.append("--" + option.replace("_", "-"))
                for option in ("max_discovery_per_class", "max_test_per_class"):
                    if getattr(args, option) is not None:
                        command.extend(["--" + option.replace("_", "-"), str(getattr(args, option))])
                jobs.append({"family": family, "model": model, "case": case, "shot": shot, "seed": seed,
                             "script": str(script), "directory": str(directory), "command": command})
    return jobs


def dataset_signature(root):
    digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        for name in ("F0", "F1", "F2", "F3", "F4"):
            folder = root / split / name
            samples = sorted(folder.glob("*.csv"))
            if not samples:
                raise FileNotFoundError(f"No CSV samples in {folder}; see data/README.md.")
            for path in samples:
                stat = path.stat()
                digest.update(f"{path.relative_to(root).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def identity(job, code_hash, data_hash):
    payload = {"command": job["command"], "code_sha256": code_hash, "dataset_inventory_sha256": data_hash}
    payload["signature"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def completed(job, request):
    directory = Path(job["directory"])
    marker = directory / "run_request.json"
    if not marker.is_file():
        return False
    previous = json.loads(marker.read_text(encoding="utf-8"))
    if previous.get("signature") != request["signature"]:
        raise ValueError(f"Configuration/code/data changed in {directory}. Select a new --output-root.")
    return (directory / "completed.json").is_file() and (directory / "summary.json").is_file()


def run_job(job, request, resume):
    directory = Path(job["directory"])
    if completed(job, request):
        if resume:
            print(f"[resume] {directory}", flush=True)
            return
        raise FileExistsError(f"Completed run already exists: {directory}. Use --resume or a new --output-root.")
    if (directory / "summary.json").exists() and not (directory / "run_request.json").exists():
        raise FileExistsError(f"Unmanaged results already exist in {directory}; choose another output root.")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env.setdefault("OMP_NUM_THREADS", "4")
    with (directory / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(job["command"], cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", env=env)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            status = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if status:
        raise subprocess.CalledProcessError(status, job["command"])
    if not (directory / "summary.json").is_file():
        raise RuntimeError(f"Run exited without summary.json: {directory}")
    (directory / "completed.json").write_text(json.dumps({"signature": request["signature"]}), encoding="utf-8")


def aggregate_jobs(jobs, output_root):
    groups = {}
    per_seed = []
    for job in jobs:
        summary = json.loads((Path(job["directory"]) / "summary.json").read_text(encoding="utf-8"))
        aliases = {"specific_class_accuracy": "accuracy", "unknown_specific_macro_recall": "unknown_macro_recall",
                   "unknown_recall_F2": "unknown_macro_recall"}
        metrics = {aliases.get(key, key): value for key, value in summary["metrics"].items()}
        metrics.update({f"unknown_{key}": value for key, value in summary.get("unknown_clustering_metrics", {}).items()})
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            row = {key: job[key] for key in ("family", "model", "case", "shot", "seed")}
            row.update(metric=metric, value=value)
            per_seed.append(row)
            key = tuple(job[k] for k in ("family", "model", "case", "shot")) + (metric,)
            groups.setdefault(key, []).append(float(value))
    rows = []
    for key, values in sorted(groups.items()):
        row = dict(zip(("family", "model", "case", "shot", "metric"), key))
        row.update(mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else 0.0, n=len(values))
        rows.append(row)
    output_root.mkdir(parents=True, exist_ok=True)
    for filename, records, keys in (
        ("repeat_metrics.csv", per_seed, ["family", "model", "case", "shot", "seed", "metric", "value"]),
        ("aggregate_metrics.csv", rows, ["family", "model", "case", "shot", "metric", "mean", "std", "n"]),
    ):
        with (output_root / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
    (output_root / "aggregate_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def main(argv=None):
    args = parse_args(argv)
    jobs = make_plan(args)
    if args.dry_run:
        for job in jobs:
            print(json.dumps({k: job[k] for k in ("family", "model", "case", "shot", "seed", "command")}, ensure_ascii=False))
        print(f"Planned {len(jobs)} independent training runs; no files written.")
        return
    data_hash = dataset_signature(args.dataset_root)
    code_hash = source_digest()
    for index, job in enumerate(jobs, 1):
        print(f"\n[{index}/{len(jobs)}] {job['model']} {job['shot']}-shot {job['case']} seed={job['seed']}", flush=True)
        run_job(job, identity(job, code_hash, data_hash), args.resume)
    aggregate_jobs(jobs, args.output_root)
    print(f"Mean and sample std (ddof=1) saved to {args.output_root / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
