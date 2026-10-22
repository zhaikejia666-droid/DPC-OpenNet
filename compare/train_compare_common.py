from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from compare_models import CompareConfig, build_model


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from release_support import default_dataset_root, configure_runtime, resolve_device, skip_plots, write_run_metadata, write_sample_manifest

ALL_CLASSES = ["F0", "F1", "F2", "F3", "F4"]
DEFAULT_SEEDS = [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]


DATASET_ROOT = default_dataset_root()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_csv_vector(path: Path) -> np.ndarray:
    values: List[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            values.extend(float(item) for item in row if item.strip())
    if not values:
        raise ValueError(f"Empty sample: {path}")
    return np.asarray(values, dtype=np.float32)


def preprocess(x: np.ndarray, target_length: int) -> np.ndarray:
    x = (x - x.mean()) / max(float(x.std()), 1e-8)
    if len(x) != target_length:
        old = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
        new = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
        x = np.interp(new, old, x)
    return x.astype(np.float32)


def files(root: Path, split: str, class_name: str) -> List[Path]:
    folder = root / split / class_name
    if not folder.exists():
        raise FileNotFoundError(f"Missing directory: {folder}")
    return sorted(folder.glob("*.csv"), key=lambda path: (len(path.stem), path.stem))


def pool_samples(
    root: Path,
    split: str,
    class_names: Sequence[str],
    max_per_class: Optional[int],
    seed: int,
) -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []
    for index, class_name in enumerate(class_names):
        class_files = files(root, split, class_name)
        rng = random.Random(seed + 1009 * (index + 1))
        rng.shuffle(class_files)
        if max_per_class is not None:
            class_files = class_files[:max_per_class]
        samples.extend((path, class_name) for path in class_files)
    return samples


class FaultDataset(Dataset):
    def __init__(self, samples, label_map, input_length: int):
        self.samples = list(samples)
        self.label_map = label_map
        self.input_length = input_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, class_name = self.samples[index]
        x = preprocess(read_csv_vector(path), self.input_length)
        return torch.from_numpy(x), torch.tensor(self.label_map[class_name], dtype=torch.long)


def augment(x: Tensor) -> Tensor:
    scale = torch.empty(x.size(0), 1, device=x.device).uniform_(0.94, 1.06)
    noise = torch.empty(x.size(0), 1, device=x.device).uniform_(0.004, 0.018)
    y = x * scale + torch.randn_like(x) * noise
    max_shift = max(1, x.size(-1) // 64)
    shifts = torch.randint(-max_shift, max_shift + 1, (x.size(0),), device=x.device)
    return torch.stack([torch.roll(sample, int(shift.item()), dims=-1) for sample, shift in zip(y, shifts)])


def model_loss(output: Dict[str, Tensor], x: Tensor, labels: Tensor, recon_weight: float, kl_weight: float):
    logits = output["logits"]
    recon = output["reconstruction"]
    if recon.ndim == 3:
        recon_flat = recon.squeeze(1)
    else:
        recon_flat = recon
    ce = F.cross_entropy(logits, labels, label_smoothing=0.04)
    recon_loss = F.mse_loss(recon_flat, x)
    loss = ce + recon_weight * recon_loss
    if "kl_loss" in output:
        loss = loss + kl_weight * output["kl_loss"]
    if "regularization_loss" in output:
        loss = loss + output["regularization_loss"]
    return loss, {"loss": float(loss.detach().cpu()), "ce": float(ce.detach().cpu()), "recon": float(recon_loss.detach().cpu())}


def train_model(model, loader, args, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "ce": 0.0, "recon": 0.0}
        correct = 0
        total = 0
        for x, labels in loader:
            x = x.to(device)
            labels = labels.to(device)
            x_aug = augment(x)
            output = model(x_aug)
            loss, parts = model_loss(output, x, labels, args.recon_weight, args.kl_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch = x.size(0)
            for key in totals:
                totals[key] += parts[key] * batch
            correct += int(output["logits"].argmax(dim=1).eq(labels).sum().detach().cpu())
            total += batch
        scheduler.step()
        row = {key: value / max(total, 1) for key, value in totals.items()}
        row["epoch"] = epoch
        row["train_acc"] = correct / max(total, 1)
        history.append(row)
        if epoch == 1 or epoch % max(1, args.print_every) == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d}/{args.epochs} loss={row['loss']:.4f} "
                f"ce={row['ce']:.4f} recon={row['recon']:.4f} acc={row['train_acc']:.4f}"
            )
    return history


class MSDAWCAEOutputAdapter(nn.Module):
    """Expose the original MS-DAWCAE outputs through the common evaluator API."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        output = self.model(x)
        z = output["z"]
        return {
            "embedding": F.normalize(z, dim=1),
            "logits": output["logits"],
            "reconstruction": output["recon"],
        }


def load_ms_dawcae_module():
    model_path = SCRIPT_DIR / "MS_DAWCAE.py"
    spec = importlib.util.spec_from_file_location("hmcvt_compare_ms_dawcae", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import MS-DAWCAE from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def train_ms_dawcae(model, module, loader, args, device):
    """Run the model's original dual-discriminator adversarial optimization."""

    ae_optimizer, d_optimizer = module.make_ms_dawcae_optimizers(model, args.lr)
    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        values = module.train_ms_dawcae_epoch(model, loader, ae_optimizer, d_optimizer, device)
        row = {
            "epoch": epoch,
            "loss": values["ae_loss"] + values["d_loss"],
            "ce": values["cls_loss"],
            "recon": values["recon_loss"],
            "train_acc": float("nan"),
            **values,
        }
        history.append(row)
        if epoch == 1 or epoch % max(1, args.print_every) == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d}/{args.epochs} ae={values['ae_loss']:.4f} "
                f"d={values['d_loss']:.4f} cls={values['cls_loss']:.4f} "
                f"recon={values['recon_loss']:.4f}"
            )
    return history


@torch.no_grad()
def extract(model, loader, device, recon_score_weight: float):
    model.eval()
    embeddings: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    pred_local: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    max_probs: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        logits = out["logits"]
        prob = F.softmax(logits, dim=1)
        max_prob, pred = prob.max(dim=1)
        recon = out["reconstruction"].squeeze(1) if out["reconstruction"].ndim == 3 else out["reconstruction"]
        recon_mse = F.mse_loss(recon, x, reduction="none").mean(dim=1)
        score = (1.0 - max_prob) + recon_score_weight * torch.log1p(recon_mse)
        embeddings.append(out["embedding"].detach().cpu().numpy())
        scores.append(score.detach().cpu().numpy())
        pred_local.append(pred.detach().cpu().numpy())
        labels.append(y.numpy())
        max_probs.append(max_prob.detach().cpu().numpy())
    return {
        "embedding": np.concatenate(embeddings, axis=0),
        "score": np.concatenate(scores, axis=0),
        "pred_local": np.concatenate(pred_local, axis=0),
        "label": np.concatenate(labels, axis=0),
        "max_prob": np.concatenate(max_probs, axis=0),
    }


def otsu_threshold(scores: np.ndarray, bins: int = 128) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) < 4 or np.allclose(scores.min(), scores.max()):
        return float(np.percentile(scores, 95.0))
    hist, edges = np.histogram(scores, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(weight2[::-1], 1))[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if len(variance12) == 0:
        return float(np.percentile(scores, 95.0))
    return float(centers[:-1][int(np.argmax(variance12))])


def adaptive_threshold(train_scores, val_scores, known_percentile: float, val_cap_percentile: float):
    known_support = float(np.percentile(train_scores, known_percentile))
    val_otsu = otsu_threshold(val_scores)
    val_cap = float(np.percentile(val_scores, val_cap_percentile))
    threshold = min(max(known_support, val_otsu), val_cap)
    return threshold, {
        "strategy": "known-train percentile + unlabeled-val Otsu threshold",
        "known_threshold_percentile": known_percentile,
        "known_train_threshold": known_support,
        "unlabeled_val_otsu_threshold": val_otsu,
        "val_threshold_cap_percentile": val_cap_percentile,
        "unlabeled_val_cap_threshold": val_cap,
        "final_global_threshold": float(threshold),
    }


def classification_metrics(matrix: np.ndarray):
    total = matrix.sum()
    accuracy = float(np.trace(matrix) / max(total, 1))
    precision = np.diag(matrix) / np.maximum(matrix.sum(axis=0), 1)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "known_macro_recall": float(recall[[0, 1, 3, 4]].mean()),
        "unknown_recall_F2": float(recall[2]),
    }


def plot_confusion(matrix: np.ndarray, class_names: Sequence[str], path: Path):
    if skip_plots():
        return
    cmap = LinearSegmentedColormap.from_list("paper_purple", ["#fbfbfd", "#d8c9ef", "#7b3294", "#3f007d"])
    row_sum = matrix.sum(axis=1, keepdims=True)
    norm = matrix / np.maximum(row_sum, 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=220)
    im = ax.imshow(norm, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predict label", fontweight="bold")
    ax.set_ylabel("Truth label", fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{norm[i, j] * 100:.1f}%\n{matrix[i, j]}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def two_dimensional_embedding(embeddings: np.ndarray, labels: np.ndarray, seed: int):
    if len(embeddings) < 4:
        return embeddings[:, :2] if embeddings.shape[1] >= 2 else np.pad(embeddings, ((0, 0), (0, 2 - embeddings.shape[1])))
    try:
        from sklearn.manifold import TSNE

        perplexity = max(2, min(30, len(embeddings) // 5))
        return TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=seed).fit_transform(embeddings)
    except Exception:
        centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[:2].T


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, class_names: Sequence[str], path: Path, seed: int):
    if skip_plots():
        return
    coords = two_dimensional_embedding(embeddings, labels, seed)
    colors = ["#1f77b4", "#009E3D", "#ff7f0e", "#9467bd", "#d62728"]
    fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=220)
    for index, name in enumerate(class_names):
        mask = labels == index
        if np.any(mask):
            ax.scatter(coords[mask, 0], coords[mask, 1], s=16, alpha=0.86, label=name, c=colors[index % len(colors)])
    ax.set_xlabel("t-SNE 1", fontweight="bold")
    ax.set_ylabel("t-SNE 2", fontweight="bold")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_history(history: List[Dict[str, float]], path: Path):
    if skip_plots():
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=200)
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["loss"] for row in history], label="loss")
    axes[0].plot(epochs, [row["ce"] for row in history], label="ce")
    axes[0].plot(epochs, [row["recon"] for row in history], label="recon")
    axes[0].legend()
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].plot(epochs, [row["train_acc"] for row in history], color="#3f007d")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train accuracy")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_predictions(path: Path, true_labels, pred_labels, scores, max_probs):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", "pred_label", "open_score", "max_prob"])
        for true, pred, score, prob in zip(true_labels, pred_labels, scores, max_probs):
            writer.writerow([ALL_CLASSES[int(true)], ALL_CLASSES[int(pred)], float(score), float(prob)])


def canonical_model_name(name: str) -> str:
    return name.strip().upper().replace("_", "-")


def build_experiment_model(args, num_classes: int, device: torch.device):
    if canonical_model_name(args.model_name) == "MS-DAWCAE":
        if args.latent_dim % 64 != 0:
            raise ValueError("MS-DAWCAE requires --latent-dim to be divisible by 64.")
        module = load_ms_dawcae_module()
        config = module.MSDAWCAEConfig(
            input_length=args.input_length,
            num_classes=num_classes,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            recon_weight=args.recon_weight,
            device=str(device),
        )
        raw_model = module.MSDAWCAE(config).to(device)
        return MSDAWCAEOutputAdapter(raw_model), raw_model, config.__dict__, module

    config = CompareConfig(
        input_length=args.input_length,
        num_classes=num_classes,
        latent_dim=args.latent_dim,
    )
    model = build_model(args.model_name, config).to(device)
    return model, model, config.__dict__, None


def run_single_experiment(args, output_dir: Path) -> Dict[str, Any]:
    if args.unknown_classes != ["F2"]:
        raise ValueError("This compare script family is configured for F2_unknown experiments.")

    set_seed(args.seed)
    configure_runtime(args)
    device = resolve_device(args.device)
    known_classes = [name for name in ALL_CLASSES if name not in args.unknown_classes]
    known_label_map = {name: index for index, name in enumerate(known_classes)}
    all_label_map = {name: index for index, name in enumerate(ALL_CLASSES)}
    local_to_global = {local: all_label_map[name] for name, local in known_label_map.items()}
    unknown_global = all_label_map["F2"]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(output_dir, args)

    train_samples = pool_samples(args.dataset_root, "train", known_classes, args.shot, args.seed)
    val_samples = pool_samples(args.dataset_root, "val", ALL_CLASSES, args.max_discovery_per_class, args.seed + 17)
    test_samples = pool_samples(args.dataset_root, "test", ALL_CLASSES, args.max_test_per_class, args.seed + 29)
    write_sample_manifest(output_dir, args.dataset_root, train=train_samples, val=val_samples, test=test_samples)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        FaultDataset(train_samples, known_label_map, args.input_length),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        generator=generator,
    )
    train_eval_loader = DataLoader(
        FaultDataset(train_samples, known_label_map, args.input_length),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        FaultDataset(val_samples, all_label_map, args.input_length),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        FaultDataset(test_samples, all_label_map, args.input_length),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    eval_model, checkpoint_model, config_dict, ms_module = build_experiment_model(args, len(known_classes), device)
    print(
        f"Model={args.model_name}, shot={args.shot}, seed={args.seed}, "
        f"device={device}, train_samples={len(train_samples)}"
    )
    if ms_module is None:
        history = train_model(eval_model, train_loader, args, device)
    else:
        history = train_ms_dawcae(checkpoint_model, ms_module, train_loader, args, device)

    train_data = extract(eval_model, train_eval_loader, device, args.recon_score_weight)
    val_data = extract(eval_model, val_loader, device, args.recon_score_weight)
    threshold, threshold_details = adaptive_threshold(
        train_data["score"],
        val_data["score"],
        args.known_threshold_percentile,
        args.val_threshold_cap_percentile,
    )
    val_rejected = val_data["score"] > threshold
    if not np.any(val_rejected):
        top_k = max(1, min(8, len(val_data["score"])))
        indices = np.argsort(val_data["score"])[-top_k:]
    else:
        indices = np.where(val_rejected)[0]
    unknown_prototype = val_data["embedding"][indices].mean(axis=0)
    unknown_prototype = unknown_prototype / max(float(np.linalg.norm(unknown_prototype)), 1e-12)

    test_data = extract(eval_model, test_loader, device, args.recon_score_weight)
    rejected = test_data["score"] > threshold
    pred_global = np.array([local_to_global[int(item)] for item in test_data["pred_local"]], dtype=int)
    pred_global[rejected] = unknown_global
    true_global = test_data["label"].astype(int)
    matrix = np.zeros((len(ALL_CLASSES), len(ALL_CLASSES)), dtype=int)
    for true, pred in zip(true_global, pred_global):
        matrix[int(true), int(pred)] += 1
    metrics = classification_metrics(matrix)

    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    write_predictions(output_dir / "predictions.csv", true_global, pred_global, test_data["score"], test_data["max_prob"])
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    plot_confusion(matrix, ALL_CLASSES, output_dir / "confusion_matrix.png")
    plot_tsne(test_data["embedding"], true_global, ALL_CLASSES, output_dir / "tsne_true_label.png", args.seed)
    plot_tsne(test_data["embedding"], pred_global, ALL_CLASSES, output_dir / "tsne_pred_label.png", args.seed)
    plot_history(history, output_dir / "training_curve.png")
    torch.save(
        {
            "model_state": checkpoint_model.state_dict(),
            "config": config_dict,
            "known_classes": known_classes,
            "threshold": threshold,
            "unknown_prototype": unknown_prototype,
        },
        output_dir / "checkpoint.pt",
    )

    summary = {
        "model_name": args.model_name,
        "shot": args.shot,
        "unknown_classes": args.unknown_classes,
        "known_classes": known_classes,
        "input_length": args.input_length,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "sample_counts": {
            "train": {name: sum(1 for _, cls in train_samples if cls == name) for name in known_classes},
            "val": {name: sum(1 for _, cls in val_samples if cls == name) for name in ALL_CLASSES},
            "test": {name: sum(1 for _, cls in test_samples if cls == name) for name in ALL_CLASSES},
        },
        "threshold_calibration": threshold_details,
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
        "notes": (
            "F2 is excluded from supervised training. Validation is an unlabeled threshold calibration pool; "
            "test is used only once for final evaluation."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print("Final metrics:", ", ".join(f"{key}={value:.6f}" for key, value in metrics.items()))
    print(f"Threshold={threshold:.6f}; results saved to {output_dir}")
    del eval_model, checkpoint_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def aggregate_runs(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    metric_names = sorted({name for row in rows for name in row["metrics"]})
    result: Dict[str, Dict[str, float]] = {}
    for name in metric_names:
        values = np.asarray([row["metrics"][name] for row in rows], dtype=np.float64)
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n": int(len(values)),
        }
    return result


def save_repeat_report(output_root: Path, args, rows: List[Dict[str, Any]]) -> None:
    aggregate = aggregate_runs(rows)
    with (output_root / "repeat_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["seed", *sorted(rows[0]["metrics"])]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"seed": row["seed"], **row["metrics"]})
    with (output_root / "aggregate_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std", "n"])
        for name, values in aggregate.items():
            writer.writerow([name, f"{values['mean']:.10f}", f"{values['std']:.10f}", values["n"]])
    report = {
        "model_name": args.model_name,
        "shot": args.shot,
        "seeds": [row["seed"] for row in rows],
        "runs": [{"seed": row["seed"], "metrics": row["metrics"], "run_dir": row["run_dir"]} for row in rows],
        "aggregate": aggregate,
        "std_definition": "sample standard deviation (ddof=1)",
    }
    with (output_root / "repeat_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print("\nAggregate metrics: mean +/- std")
    for name, values in aggregate.items():
        print(f"{name}: {values['mean']:.6f} +/- {values['std']:.6f} (n={values['n']})")


def run_experiment(default_model: str, default_shot: int):
    parser = argparse.ArgumentParser(description="F2-unknown comparison with ten independent seeded runs.")
    parser.add_argument("--model-name", default=default_model)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None, help="Root containing seed_<seed> subdirectories.")
    parser.add_argument("--unknown-classes", nargs="+", default=["F2"])
    parser.add_argument("--shot", type=int, default=default_shot)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None, help="Default: 3e-4; MS-DAWCAE uses its paper setting 1e-5.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-length", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--recon-weight", type=float, default=None)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--recon-score-weight", type=float, default=0.05)
    parser.add_argument("--known-threshold-percentile", type=float, default=95.0)
    parser.add_argument("--val-threshold-cap-percentile", type=float, default=80.0)
    parser.add_argument("--max-discovery-per-class", type=int, default=None)
    parser.add_argument("--max-test-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026, help="Used only with --single-run.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--single-run", action="store_true", help="Run only --seed; useful for debugging.")
    parser.add_argument("--resume", action="store_true", help="Reuse seed folders containing summary.json.")
    parser.add_argument("--dry-run", action="store_true", help="Write the planned seeds without training.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    is_ms_dawcae = canonical_model_name(args.model_name) == "MS-DAWCAE"
    if args.lr is None:
        args.lr = 1e-5 if is_ms_dawcae else 3e-4
    if args.recon_weight is None:
        args.recon_weight = 1.0 if is_ms_dawcae else 0.20

    safe_model = args.model_name.replace("-", "_")
    output_root = args.output_dir or PROJECT_DIR / "results" / "baselines" / f"{safe_model}_{args.shot}shot_F2_unknown_10seed_results"
    output_root.mkdir(parents=True, exist_ok=True)
    if args.single_run:
        return run_single_experiment(args, output_root)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain unique values for independent repeats.")
    if args.dry_run:
        plan = {"model_name": args.model_name, "shot": args.shot, "seeds": args.seeds, "output_root": str(output_root)}
        with (output_root / "repeat_plan.json").open("w", encoding="utf-8") as handle:
            json.dump(plan, handle, ensure_ascii=False, indent=2)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan

    rows: List[Dict[str, Any]] = []
    for index, seed in enumerate(args.seeds, start=1):
        run_dir = output_root / f"seed_{seed}"
        summary_path = run_dir / "summary.json"
        print(f"\n[{index}/{len(args.seeds)}] seed={seed}")
        if args.resume and summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            print(f"[resume] loaded {summary_path}")
        else:
            args.seed = seed
            summary = run_single_experiment(args, run_dir)
        rows.append({"seed": seed, "metrics": summary["metrics"], "run_dir": str(run_dir)})
    save_repeat_report(output_root, args, rows)
    return rows


if __name__ == "__main__":
    run_experiment("FEUG", 20)
