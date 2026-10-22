"""Checks for relocation, model interfaces, label-free calibration and aggregation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

import numpy as np
import torch

from models.dpc_opennet import DPCOpenNet, DPCOpenNetConfig
from release_support import ROOT
from run_experiments import aggregate_jobs, completed, make_plan, parse_args


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_proposed_entries_import_without_dataset(self):
        entries = sorted((ROOT / "Experiment").glob("*shot_*_unknown.py"))
        self.assertEqual(len(entries), 9)
        for index, path in enumerate(entries):
            with self.subTest(script=path.name):
                module = load(path, f"entry_{index}")
                self.assertTrue(module.MODEL_FILE.is_file())
                self.assertEqual(module.PROJECT_DIR, ROOT)

    def test_paper_component_names(self):
        self.assertEqual(DPCOpenNet.__name__, "DPCOpenNet")
        model = DPCOpenNet(DPCOpenNetConfig(device="cpu"))
        self.assertEqual([type(module).__name__ for module in model.children()], [
            "MultiscaleDualViewAdaptiveEncoding",
            "PrototypeContrastiveKnownFaultLearning",
            "ReconstructionAndDualSpaceAdversarialRegularization",
            "ResidualAndSpectralRepresentations",
        ])

    def test_previous_checkpoint_parameter_names_load(self):
        first = DPCOpenNet(DPCOpenNetConfig(device="cpu")).eval()
        previous = {}
        for key, value in first.state_dict().items():
            for old, new in first._CHECKPOINT_PREFIXES.items():
                if key.startswith(new + "."):
                    key = old + key[len(new):]
                    break
            previous[key] = value
        second = DPCOpenNet(DPCOpenNetConfig(device="cpu")).eval()
        second.load_state_dict(previous, strict=True)
        signal = torch.randn(2, 1, 256)
        with torch.no_grad():
            torch.testing.assert_close(first(signal)["logits"], second(signal)["logits"], rtol=0, atol=0)

    def test_experiment_plan_has_360_unique_runs(self):
        jobs = make_plan(parse_args(["--method", "all"]))
        self.assertEqual(len(jobs), 360)
        self.assertEqual(len({job["directory"] for job in jobs}), 360)

    def test_known_classes_exclude_unknown_training_labels(self):
        protocol = load(ROOT / "Experiment" / "5shot_F2_unknown.py", "single_protocol")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("F0", "F1", "F3", "F4"):
                folder = root / "train" / name
                folder.mkdir(parents=True)
                for i in range(9):
                    (folder / f"{i}.csv").touch()
            fit, calibration = protocol.split_known_train(root, ["F0", "F1", "F3", "F4"], 0, 2026, 5)
            self.assertEqual(len(fit), 20)
            self.assertEqual(calibration, [])
            self.assertNotIn("F2", [name for _, name in fit])

    def test_multi_calibration_requires_no_unlabeled_ground_truth(self):
        protocol = load(ROOT / "Experiment" / "5shot_F2_F3_unknown.py", "multi_protocol")
        fit_scores = np.linspace(0, 2, 20)
        val_scores = np.linspace(0, 5, 40)
        fit = {"true_label": np.repeat([0, 1], 10), "confidence": np.full(20, 0.85), "margin": np.full(20, 0.7)}
        val = {"known_prediction": np.tile([0, 1], 20)}
        first = protocol.adaptive_fewshot_thresholds(fit_scores, val_scores, fit, val, 2, 95, 80)
        val["true_label"] = np.full(40, -999)
        second = protocol.adaptive_fewshot_thresholds(fit_scores, val_scores, fit, val, 2, 95, 80)
        self.assertEqual(first, second)

    def test_proposed_shapes_and_gradients(self):
        for count in (2, 3, 4):
            with self.subTest(known_classes=count):
                model = DPCOpenNet(DPCOpenNetConfig(num_known_classes=count, device="cpu"))
                output = model(torch.randn(2, 1, 256))
                self.assertEqual(output["logits"].shape, (2, count))
                self.assertEqual(output["recon"].shape, (2, 1, 256))
                self.assertEqual(output["spectral_embedding"].shape, (2, 32))
                loss = output["recon"].square().mean() + output["logits"].square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertIsNotNone(next(model.adaptive_encoding.dual_view_branches[0].parameters()).grad)

    def test_all_eight_comparison_models_forward_backward(self):
        module = load(ROOT / "compare" / "compare_models.py", "compare_models_test")
        for name in ("CRFDE", "DVPN", "GEBS", "OFER", "PDAgg", "SSO-FC", "TNPNet", "FEUG"):
            with self.subTest(model=name):
                model = module.build_model(name, module.CompareConfig())
                output = model(torch.randn(4, 256))
                self.assertEqual(output["logits"].shape, (4, 4))
                self.assertEqual(output["reconstruction"].shape, (4, 1, 256))
                loss = output["logits"].square().mean() + output["reconstruction"].square().mean()
                loss = loss + output.get("regularization_loss", 0) + output.get("kl_loss", 0)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))

    def test_original_ms_dawcae_forward_backward(self):
        module = load(ROOT / "compare" / "MS_DAWCAE.py", "original_model_test")
        model = module.MSDAWCAE(module.MSDAWCAEConfig(input_length=256, latent_dim=128, num_classes=4, device="cpu"))
        output = model(torch.randn(2, 256))
        self.assertEqual(output["logits"].shape, (2, 4))
        self.assertEqual(output["recon"].shape, (2, 1, 256))
        loss = output["recon"].square().mean() + output["logits"].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_model_is_a_single_standalone_file(self):
        self.assertEqual([path.name for path in (ROOT / "models").glob("*.py")], ["dpc_opennet.py"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "standalone.py"
            shutil.copy2(ROOT / "models" / "dpc_opennet.py", path)
            module = load(path, "standalone_model_test")
            model = module.DPCOpenNet(module.DPCOpenNetConfig(num_known_classes=2, device="cpu")).eval()
            with torch.no_grad():
                output = model(torch.randn(2, 1, 256))
            self.assertEqual(output["logits"].shape, (2, 2))

    def test_aggregate_uses_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = []
            for seed, value in ((2026, 0.6), (2027, 0.8)):
                folder = root / str(seed)
                folder.mkdir()
                (folder / "summary.json").write_text(json.dumps({"metrics": {"accuracy": value}}))
                jobs.append(dict(family="proposed", model="dpc_opennet", case="single", shot=5, seed=seed, directory=str(folder)))
            rows = aggregate_jobs(jobs, root / "report")
            self.assertAlmostEqual(rows[0]["mean"], 0.7)
            self.assertAlmostEqual(rows[0]["std"], 2**0.5 / 10)
            self.assertEqual(rows[0]["n"], 2)

    def test_resume_rejects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_request.json").write_text(json.dumps({"signature": "old"}))
            with self.assertRaises(ValueError):
                completed({"directory": directory}, {"signature": "new"})


if __name__ == "__main__":
    unittest.main()
