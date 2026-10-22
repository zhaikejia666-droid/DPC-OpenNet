"""Portable paths and execution metadata for the released experiment scripts."""

from __future__ import annotations

import hashlib
import csv
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def default_dataset_root() -> Path:
    configured = os.environ.get("DPC_DATASET_ROOT")
    return Path(configured).expanduser() if configured else ROOT / "data" / "dataset_8000"


def source_digest() -> str:
    digest = hashlib.sha256()
    files = [*ROOT.glob("*.py")]
    for folder in ("models", "Experiment", "compare"):
        files.extend((ROOT / folder).glob("*.py"))
    for path in sorted(files):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_device(request: str):
    import torch

    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(request)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use --device auto, cpu, cuda, or cuda:<index>.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable in this environment.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is out of range: {request}")
    return device


def configure_runtime(args) -> None:
    import torch

    if args.num_threads < 1 or args.epochs < 1 or args.batch_size < 2:
        raise ValueError("Require num_threads >= 1, epochs >= 1, and batch_size >= 2.")
    # Five fixed scales and the inherited decoder constrain these dimensions.
    if args.input_length < 128 or args.input_length % 32:
        raise ValueError("--input-length must be >= 128 and divisible by 32; use 256 for the paper protocol.")
    if args.latent_dim < 64 or args.latent_dim % 64:
        raise ValueError("--latent-dim must be a positive multiple of 64.")
    if not 0 <= args.known_threshold_percentile <= 100:
        raise ValueError("Known threshold percentile must be within [0, 100].")
    if not 0 <= args.val_threshold_cap_percentile <= 100:
        raise ValueError("Validation cap percentile must be within [0, 100].")
    torch.set_num_threads(args.num_threads)
    resolve_device(args.device)
    os.environ["DPC_SKIP_PLOTS"] = "1" if args.skip_plots else "0"
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def skip_plots() -> bool:
    return os.environ.get("DPC_SKIP_PLOTS") == "1"


def write_run_metadata(output_dir: Path, args) -> None:
    import torch

    versions = {}
    for package in ("torch", "numpy", "scipy", "scikit-learn", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    device = resolve_device(args.device)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_version": torch.version.cuda,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "code_sha256": source_digest(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_sample_manifest(output_dir, dataset_root, **partitions):
    root = Path(dataset_root).resolve()
    with (Path(output_dir) / "sample_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["role", "relative_file", "class_for_evaluation"])
        for role, samples in partitions.items():
            for path, name in samples:
                writer.writerow([role, Path(path).resolve().relative_to(root).as_posix(), name])
