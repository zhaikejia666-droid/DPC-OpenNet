"""Print the runtime versions needed when reporting reproducible experiments."""

import importlib.metadata
import json
import platform
import sys

import torch


def main():
    report = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    for name in ("torch", "numpy", "scipy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl"):
        report["packages"][name] = importlib.metadata.version(name)
    report.update(cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda)
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
