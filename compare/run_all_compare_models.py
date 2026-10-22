"""Run every model at 5/10/20 shot with ten independent seeds."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_STEMS = ["crfde", "dvpn", "gebs", "ofer", "pdagg", "sso_fc", "tnpnet", "feug", "ms_dawcae"]
SHOTS = [5, 10, 20]


def experiment_scripts(models: List[str], shots: List[int]) -> List[Path]:
    scripts = [SCRIPT_DIR / f"train_{model}_{shot}shot_F2_unknown.py" for model in models for shot in shots]
    missing = [path for path in scripts if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing experiment scripts:\n" + "\n".join(str(path) for path in missing))
    return scripts


def default_result_dir(script: Path) -> Path:
    experiment = script.stem.removeprefix("train_")
    parts = experiment.split("_")
    shot_index = next(index for index, item in enumerate(parts) if item.endswith("shot"))
    model = "_".join(parts[:shot_index])
    shot = parts[shot_index]
    return PROJECT_DIR / "results" / "baselines" / f"{model}_{shot}_F2_unknown_10seed_results"


def collect_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_combined_reports(output_dir: Path, reports: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "all_compare_repeat_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, ensure_ascii=False, indent=2)
    rows = []
    for experiment, report in reports.items():
        for metric, values in report.get("aggregate", {}).items():
            rows.append(
                {
                    "experiment": experiment,
                    "model": report.get("model_name"),
                    "shot": report.get("shot"),
                    "metric": metric,
                    "mean": values.get("mean"),
                    "std": values.get("std"),
                    "n": values.get("n"),
                }
            )
    with (output_dir / "all_compare_aggregate_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "model", "shot", "metric", "mean", "std", "n"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all 9 models x 3 shot settings x 10 seeds.")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--models", nargs="+", choices=MODEL_STEMS, default=MODEL_STEMS)
    parser.add_argument("--shots", type=int, nargs="+", choices=SHOTS, default=SHOTS)
    parser.add_argument("--output-parent", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded = parser.parse_known_args()

    scripts = experiment_scripts(args.models, args.shots)
    result_locations: Dict[str, Path] = {}
    for index, script in enumerate(scripts, start=1):
        result_dir = (
            args.output_parent / (script.stem + "_10seed_results")
            if args.output_parent is not None
            else default_result_dir(script)
        )
        command = [args.python_executable, str(script), "--output-dir", str(result_dir)]
        if args.resume:
            command.append("--resume")
        if args.dry_run:
            command.append("--dry-run")
        command.extend(forwarded)
        print(f"\n[{index}/{len(scripts)}] {script.name}")
        subprocess.run(command, check=True)
        result_locations[script.stem] = result_dir

    if args.dry_run:
        print("Dry-run complete; no models were trained.")
        return

    reports: Dict[str, Any] = {}
    for experiment, result_dir in result_locations.items():
        report_path = result_dir / "repeat_summary.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing aggregate report: {report_path}")
        reports[experiment] = collect_report(report_path)
    report_dir = args.output_parent or PROJECT_DIR / "results" / "baselines"
    write_combined_reports(report_dir, reports)
    print(f"\nCombined reports saved to {report_dir}")


if __name__ == "__main__":
    main()
