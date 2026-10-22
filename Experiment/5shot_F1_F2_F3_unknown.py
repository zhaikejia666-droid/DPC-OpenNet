"""DPC-OpenNet training and evaluation: 5shot F1 F2 F3 unknown.

Train uses labeled known faults; val calibrates thresholds and unknown
prototypes; test provides the final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from release_support import default_dataset_root, configure_runtime, resolve_device, skip_plots, write_run_metadata, write_sample_manifest

DATASET_ROOT = default_dataset_root()
MODEL_FILE = PROJECT_DIR / "models" / "dpc_opennet.py"
OUTPUT_DIR = PROJECT_DIR / "results" / "proposed" / "5shot_F1_F2_F3_unknown"
ALL_AVAILABLE_CLASSES = ["F0", "F1", "F2", "F3", "F4"]


def load_model_module(path: Path):
    spec = importlib.util.spec_from_file_location("dpc_opennet_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


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


def spectral_descriptor(x: np.ndarray, points: int = 64) -> np.ndarray:
    if len(x) != points:
        old = np.linspace(0.0, 1.0, len(x), dtype=np.float64)
        new = np.linspace(0.0, 1.0, points, dtype=np.float64)
        x = np.interp(new, old, x)
    x = (x - x.mean()) / max(float(x.std()), 1e-8)
    return np.log1p(np.abs(np.fft.rfft(x))[1:]).astype(np.float32)


def files(root: Path, split: str, class_name: str) -> List[Path]:
    folder = root / split / class_name
    if not folder.exists():
        raise FileNotFoundError(f"Missing directory: {folder}")
    return sorted(folder.glob("*.csv"), key=lambda path: (len(path.stem), path.stem))


def split_known_train(
    root: Path,
    known_classes: Sequence[str],
    ratio: float,
    seed: int,
    max_per_class: Optional[int],
):
    fit: List[Tuple[Path, str]] = []
    calibration: List[Tuple[Path, str]] = []
    for index, class_name in enumerate(known_classes):
        class_files = files(root, "train", class_name)
        rng = random.Random(seed + index)
        rng.shuffle(class_files)
        if max_per_class is not None:
            class_files = class_files[:max_per_class]
        if ratio <= 0.0:
            calibration_count = 0
        else:
            calibration_count = min(
                max(1, int(round(len(class_files) * ratio))),
                max(1, len(class_files) - 1),
            )
        calibration.extend((path, class_name) for path in class_files[:calibration_count])
        fit.extend((path, class_name) for path in class_files[calibration_count:])
    return fit, calibration


def pool_samples(
    root: Path,
    split: str,
    class_names: Sequence[str],
    max_per_class: Optional[int],
    seed: int,
):
    samples: List[Tuple[Path, str]] = []
    for index, class_name in enumerate(class_names):
        class_files = files(root, split, class_name)
        if max_per_class is not None:
            rng = random.Random(seed + index)
            rng.shuffle(class_files)
            class_files = class_files[:max_per_class]
        samples.extend((path, class_name) for path in class_files)
    return samples


class FaultDataset(Dataset):
    def __init__(self, samples, label_map, input_length: int) -> None:
        self.samples = list(samples)
        self.label_map = label_map
        self.input_length = input_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, class_name = self.samples[index]
        raw = read_csv_vector(path)
        x = preprocess(raw, self.input_length)
        spectrum = spectral_descriptor(raw)
        return (
            torch.from_numpy(x),
            torch.from_numpy(spectrum),
            torch.tensor(self.label_map[class_name], dtype=torch.long),
        )


def counts(samples, class_names) -> Dict[str, int]:
    result = {name: 0 for name in class_names}
    for _, name in samples:
        result[name] += 1
    return result


def augment(x: Tensor) -> Tensor:
    scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(0.93, 1.07)
    noise = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(0.005, 0.02)
    augmented = x * scale + torch.randn_like(x) * noise
    max_shift = max(1, x.size(-1) // 64)
    shifts = torch.randint(-max_shift, max_shift + 1, (x.size(0),), device=x.device)
    return torch.stack(
        [torch.roll(sample, int(shift.item()), dims=-1) for sample, shift in zip(augmented, shifts)]
    )


def supervised_contrastive(features: Tensor, labels: Tensor, temperature: float = 0.12):
    batch, views, _ = features.shape
    features = F.normalize(features.reshape(batch * views, -1), dim=1)
    labels = labels.repeat_interleave(views)
    logits = features @ features.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(len(features), device=features.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    exp_logits = torch.exp(logits) * ~eye
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    return -(
        (positives * log_probability).sum(dim=1)
        / positives.sum(dim=1).clamp_min(1)
    ).mean()


def prototype_margin(model, z: Tensor, labels: Tensor, margin: float = 0.20):
    similarity = F.normalize(z, dim=1) @ model.known_prototypes().t()
    own = similarity.gather(1, labels[:, None]).squeeze(1)
    other = similarity.masked_fill(
        F.one_hot(labels, similarity.size(1)).bool(),
        -1e9,
    ).max(dim=1).values
    return F.relu(margin - own + other).mean()


def pseudo_unknown(z: Tensor, labels: Tensor) -> Tensor:
    permutation = torch.randperm(len(z), device=z.device)
    for _ in range(4):
        same = labels == labels[permutation]
        if not same.any():
            break
        permutation[same] = permutation[same].roll(1)
    alpha = torch.empty(len(z), 1, device=z.device).uniform_(0.35, 0.65)
    return alpha * z + (1.0 - alpha) * z[permutation] + 0.04 * torch.randn_like(z)


def boundary_exposure_loss(model, known_logits: Tensor, mixed_z: Tensor):
    mixed_logits = model.classifier(mixed_z)
    uniform = torch.full_like(mixed_logits, 1.0 / mixed_logits.size(1))
    uniform_loss = F.kl_div(
        F.log_softmax(mixed_logits, dim=1),
        uniform,
        reduction="batchmean",
    )
    energy_margin = F.relu(
        1.0 + model.energy(known_logits).detach().mean()
        - model.energy(mixed_logits).mean()
    )
    return uniform_loss + energy_margin


def set_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def train_epoch(model, loader, main_optimizer, discriminator_optimizer, device):
    model.train()
    bce = nn.BCELoss()
    totals = {
        "total": 0.0,
        "reconstruction": 0.0,
        "classification": 0.0,
        "contrastive": 0.0,
        "prototype": 0.0,
        "boundary": 0.0,
        "adversarial": 0.0,
        "discriminator": 0.0,
    }
    batches = 0

    for x, _, labels in loader:
        x = x.to(device).float().unsqueeze(1)
        labels = labels.to(device)
        view_a = augment(x)
        view_b = augment(x)
        batch = len(x)
        ones = torch.ones(batch, 1, device=device)
        zeros = torch.zeros(batch, 1, device=device)

        set_grad(model.encoder_discriminator, True)
        set_grad(model.decoder_discriminator, True)
        discriminator_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            z = model.encode(view_a)["z"]
            recon, _ = model.decode(z)
            random_z = torch.empty_like(z).uniform_(-1, 1)
            random_recon, _ = model.decode(random_z)
        discriminator_loss = (
            bce(model.encoder_discriminator(z), ones)
            + bce(model.encoder_discriminator(random_z), zeros)
            + bce(model.decoder_discriminator(recon), ones)
            + bce(model.decoder_discriminator(random_recon), zeros)
        )
        discriminator_loss.backward()
        discriminator_optimizer.step()

        set_grad(model.encoder_discriminator, False)
        set_grad(model.decoder_discriminator, False)
        main_optimizer.zero_grad(set_to_none=True)
        output_a = model(view_a)
        output_b = model(view_b)
        reconstruction = 0.5 * (
            F.smooth_l1_loss(output_a["recon"], x)
            + F.smooth_l1_loss(output_b["recon"], x)
        )
        branch_reconstruction = torch.stack(
            [F.smooth_l1_loss(branch, x) for branch in output_a["branch_recons"]]
        ).mean()
        reconstruction = 0.75 * reconstruction + 0.25 * branch_reconstruction
        classification = 0.5 * (
            F.cross_entropy(
                output_a["logits"],
                labels,
                label_smoothing=model.config.label_smoothing,
            )
            + F.cross_entropy(
                output_b["logits"],
                labels,
                label_smoothing=model.config.label_smoothing,
            )
        )
        contrastive = supervised_contrastive(
            torch.stack((output_a["projection"], output_b["projection"]), dim=1),
            labels,
        )
        prototype = 0.5 * (
            prototype_margin(model, output_a["z"], labels)
            + prototype_margin(model, output_b["z"], labels)
        )
        boundary = boundary_exposure_loss(
            model,
            output_a["logits"],
            pseudo_unknown(output_a["z"].detach(), labels),
        )
        random_z = torch.empty_like(output_a["z"]).uniform_(-1, 1)
        random_recon, _ = model.decode(random_z)
        adversarial = (
            bce(model.encoder_discriminator(output_a["z"]), ones)
            + bce(model.decoder_discriminator(output_a["recon"]), ones)
            + bce(model.decoder_discriminator(random_recon), ones)
        )
        total = (
            model.config.recon_weight * reconstruction
            + model.config.cls_weight * classification
            + model.config.contrast_weight * contrastive
            + model.config.prototype_weight * prototype
            + 0.06 * boundary
            + model.config.adversarial_weight * adversarial
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        main_optimizer.step()
        set_grad(model.encoder_discriminator, True)
        set_grad(model.decoder_discriminator, True)

        values = {
            "total": total,
            "reconstruction": reconstruction,
            "classification": classification,
            "contrastive": contrastive,
            "prototype": prototype,
            "boundary": boundary,
            "adversarial": adversarial,
            "discriminator": discriminator_loss,
        }
        for key, value in values.items():
            totals[key] += float(value.detach().cpu())
        batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def known_statistics(model, loader, device, num_classes):
    model.eval()
    all_z: List[Tensor] = []
    all_labels: List[Tensor] = []
    for x, _, labels in loader:
        output = model.infer(x.to(device).float().unsqueeze(1))
        all_z.append(output["z"].cpu())
        all_labels.append(labels)
    z = torch.cat(all_z)
    labels = torch.cat(all_labels)
    means = []
    variances = []
    for index in range(num_classes):
        class_z = z[labels == index]
        means.append(class_z.mean(dim=0))
        variances.append(class_z.var(dim=0, unbiased=False).clamp_min(1e-3))
    return torch.stack(means), torch.stack(variances)


@torch.no_grad()
def collect(model, loader, device, means, variances):
    model.eval()
    means = means.to(device)
    variances = variances.to(device)
    output_data = {
        "energy": [],
        "mahalanobis": [],
        "residual": [],
        "confidence": [],
        "margin": [],
        "known_prediction": [],
        "true_label": [],
        "discovery_embedding": [],
        "spectral_embedding": [],
        "semantic_embedding": [],
    }
    for x, spectrum, labels in loader:
        x = x.to(device).float().unsqueeze(1)
        output = model.infer(x, spectrum.to(device).float())
        distances = (
            (output["z"][:, None] - means[None]).square()
            / variances[None]
        ).mean(dim=-1)
        probabilities = F.softmax(output["logits"], dim=1)
        topk = torch.topk(probabilities, k=min(2, probabilities.size(1)), dim=1).values
        confidence = topk[:, 0]
        margin = (
            topk[:, 0] - topk[:, 1]
            if topk.size(1) > 1
            else topk[:, 0]
        )
        output_data["energy"].append(model.energy(output["logits"]).cpu().numpy())
        output_data["mahalanobis"].append(distances.min(dim=1).values.cpu().numpy())
        output_data["residual"].append(
            output["residual"].square().flatten(1).mean(dim=1).cpu().numpy()
        )
        output_data["confidence"].append(confidence.cpu().numpy())
        output_data["margin"].append(margin.cpu().numpy())
        output_data["known_prediction"].append(
            output["logits"].argmax(dim=1).cpu().numpy()
        )
        output_data["true_label"].append(labels.numpy())
        output_data["discovery_embedding"].append(
            output["discovery_embedding"].cpu().numpy()
        )
        output_data["spectral_embedding"].append(
            output["spectral_embedding"].cpu().numpy()
        )
        output_data["semantic_embedding"].append(output["projection"].cpu().numpy())
    return {key: np.concatenate(value) for key, value in output_data.items()}


def robust_stats(values: np.ndarray):
    median = float(np.median(values))
    scale = max(1.4826 * float(np.median(np.abs(values - median))), 1e-6)
    return median, scale


def fit_normalizer(data):
    return {
        key: robust_stats(data[key])
        for key in ("energy", "mahalanobis", "residual")
    }


def novelty_score(data, normalizer):
    weights = {"energy": 0.25, "mahalanobis": 0.45, "residual": 0.30}
    score = np.zeros_like(data["energy"], dtype=float)
    for key, weight in weights.items():
        center, scale = normalizer[key]
        score += weight * (data[key] - center) / scale
    return score


def class_thresholds(scores, predictions, percentile, num_classes):
    global_threshold = float(np.percentile(scores, percentile))
    thresholds = {}
    for index in range(num_classes):
        selected = scores[predictions == index]
        thresholds[index] = (
            float(np.percentile(selected, percentile))
            if len(selected) >= 10
            else global_threshold
        )
    return thresholds


def otsu_threshold(scores: np.ndarray, bins: int = 128) -> float:
    """Estimate a label-free valley threshold from the unlabeled val score histogram."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) < 8:
        return float(np.percentile(scores, 90.0)) if len(scores) else 0.0
    lower, upper = np.percentile(scores, [1.0, 99.0])
    clipped = np.clip(scores, lower, upper)
    histogram, edges = np.histogram(clipped, bins=bins)
    centers = (edges[:-1] + edges[1:]) * 0.5
    weight_left = np.cumsum(histogram).astype(float)
    weight_right = weight_left[-1] - weight_left
    valid = (weight_left > 0) & (weight_right > 0)
    if not np.any(valid):
        return float(np.percentile(scores, 90.0))
    mean_left = np.cumsum(histogram * centers) / np.maximum(weight_left, 1e-12)
    mean_right = (
        np.cumsum((histogram * centers)[::-1])
        / np.maximum(np.cumsum(histogram[::-1]), 1e-12)
    )[::-1]
    between = weight_left * weight_right * np.square(mean_left - mean_right)
    between[~valid] = -1.0
    return float(centers[int(np.argmax(between))])


def adaptive_fewshot_thresholds(
    fit_scores: np.ndarray,
    discovery_scores: np.ndarray,
    fit_data,
    discovery_data,
    num_classes: int,
    known_percentile: float,
    val_percentile_cap: float,
):
    """Combine local known support with unlabeled-val boundary evidence.

    A single global threshold misses weak unknowns that are absorbed by a nearby
    known class. This routine therefore calibrates a threshold for each predicted
    known class, and adds confidence/margin gates from the few labeled known
    samples. Ground-truth labels from val are never used here.
    """
    known_support = float(np.percentile(fit_scores, known_percentile))
    val_otsu = otsu_threshold(discovery_scores)
    val_cap = float(np.percentile(discovery_scores, val_percentile_cap))
    global_threshold = min(max(known_support, val_otsu), val_cap)
    thresholds = {}
    confidence_thresholds = {}
    margin_thresholds = {}
    class_details = {}
    fit_true = fit_data["true_label"]
    for index in range(num_classes):
        fit_mask = fit_true == index
        fit_class_scores = fit_scores[fit_mask]
        fit_class_confidence = fit_data["confidence"][fit_mask]
        fit_class_margin = fit_data["margin"][fit_mask]
        predicted_as_class = discovery_data["known_prediction"] == index
        val_class_scores = discovery_scores[predicted_as_class]
        class_known_support = (
            float(np.percentile(fit_class_scores, known_percentile))
            if len(fit_class_scores)
            else known_support
        )
        if len(val_class_scores) >= 8:
            class_otsu = otsu_threshold(val_class_scores)
            class_cap = float(np.percentile(val_class_scores, val_percentile_cap))
            class_threshold = min(max(class_known_support, class_otsu), class_cap)
        else:
            class_otsu = val_otsu
            class_cap = val_cap
            class_threshold = global_threshold
        thresholds[index] = float(class_threshold)
        confidence_thresholds[index] = (
            float(max(0.0, np.percentile(fit_class_confidence, 5.0) - 0.02))
            if len(fit_class_confidence)
            else 0.0
        )
        margin_thresholds[index] = (
            float(max(0.0, np.percentile(fit_class_margin, 5.0) * 0.80))
            if len(fit_class_margin)
            else 0.0
        )
        class_details[index] = {
            "known_support": class_known_support,
            "unlabeled_val_otsu": float(class_otsu),
            "unlabeled_val_cap": float(class_cap),
            "final_novelty_threshold": float(class_threshold),
            "confidence_lower_threshold": confidence_thresholds[index],
            "margin_lower_threshold": margin_thresholds[index],
            "fit_count": int(np.sum(fit_mask)),
            "unlabeled_val_predicted_count": int(np.sum(predicted_as_class)),
        }
    details = {
        "strategy": "class-conditional known-train support + unlabeled-val Otsu + confidence/margin weak-unknown gates",
        "known_train_percentile": known_percentile,
        "known_train_threshold": known_support,
        "unlabeled_val_otsu_threshold": val_otsu,
        "unlabeled_val_percentile_cap": val_percentile_cap,
        "unlabeled_val_cap_threshold": val_cap,
        "final_global_threshold": global_threshold,
        "class_details": class_details,
    }
    return thresholds, confidence_thresholds, margin_thresholds, details


def reject_mask(
    scores,
    predictions,
    thresholds,
    margin: float = 0.0,
    confidence=None,
    confidence_thresholds=None,
    logit_margin=None,
    margin_thresholds=None,
):
    return np.asarray(
        [
            (
                score > thresholds[int(prediction)] + margin
                or (
                    confidence is not None
                    and confidence_thresholds is not None
                    and confidence[row] < confidence_thresholds[int(prediction)]
                )
                or (
                    logit_margin is not None
                    and margin_thresholds is not None
                    and logit_margin[row] < margin_thresholds[int(prediction)]
                )
            )
            for row, (score, prediction) in enumerate(zip(scores, predictions))
        ],
        dtype=bool,
    )


def estimate_cluster_count(embeddings: np.ndarray, max_clusters: int, seed: int):
    if len(embeddings) < 6 or max_clusters <= 1:
        return 1, {}
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    maximum = min(max_clusters, max(2, len(embeddings) // 5))
    scores = {}
    best_k = 1
    best_score = -1.0
    for k in range(2, maximum + 1):
        labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(embeddings)
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(embeddings, labels, metric="euclidean"))
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    if best_score < 0.22:
        best_k = 1
    return best_k, scores


def fit_discovery_transform(embeddings: np.ndarray):
    """Robustly scale and whiten spectral descriptors before clustering."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    scaled = scaler.fit_transform(embeddings)
    components = min(24, scaled.shape[1], max(1, len(scaled) - 1))
    pca = PCA(n_components=components, whiten=True, random_state=2026)
    transformed = pca.fit_transform(scaled)
    state = {
        "center": scaler.center_.astype(np.float32),
        "scale": scaler.scale_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),
        "pca_whiten_scale": np.sqrt(
            np.maximum(pca.explained_variance_, 1e-12)
        ).astype(np.float32),
    }
    return transformed, state


def apply_discovery_transform(embeddings: np.ndarray, state):
    scaled = (
        embeddings - state["center"]
    ) / np.maximum(state["scale"], 1e-12)
    centered = scaled - state["pca_mean"]
    transformed = centered @ state["pca_components"].T
    return transformed / np.maximum(state["pca_whiten_scale"], 1e-12)


def discover_unknown_prototypes(
    embeddings: np.ndarray,
    requested_clusters: int,
    max_clusters: int,
    seed: int,
):
    from sklearn.cluster import KMeans

    if len(embeddings) == 0:
        raise RuntimeError("No unknown candidates were found in the discovery pool.")
    transformed, transform_state = fit_discovery_transform(embeddings)
    if requested_clusters > 0:
        cluster_count = min(requested_clusters, len(embeddings))
        silhouette = {}
    else:
        cluster_count, silhouette = estimate_cluster_count(
            transformed,
            max_clusters,
            seed,
        )
    if cluster_count == 1:
        labels = np.zeros(len(transformed), dtype=int)
        prototypes = transformed.mean(axis=0, keepdims=True)
    else:
        estimator = KMeans(
            n_clusters=cluster_count,
            n_init=50,
            random_state=seed,
        )
        labels = estimator.fit_predict(transformed)
        prototypes = estimator.cluster_centers_
    return labels, np.asarray(prototypes), silhouette, transform_state


def assign_unknown_clusters(embeddings, prototypes, transform_state):
    transformed = apply_discovery_transform(embeddings, transform_state)
    distances = (
        transformed[:, None, :] - prototypes[None, :, :]
    ) ** 2
    return np.argmin(distances.sum(axis=-1), axis=1)


def known_spectral_prototypes(data, transform_state, num_known_classes: int):
    transformed = apply_discovery_transform(data["spectral_embedding"], transform_state)
    labels = data["true_label"]
    fallback = transformed.mean(axis=0)
    prototypes = []
    for index in range(num_known_classes):
        selected = transformed[labels == index]
        prototypes.append(selected.mean(axis=0) if len(selected) else fallback)
    return np.asarray(prototypes, dtype=np.float32)


def nearest_distances(transformed, prototypes):
    distances = ((transformed[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=-1)
    return distances.min(axis=1), distances.argmin(axis=1), distances


def prototype_rescue_state(
    fit_data,
    candidate_embeddings,
    unknown_prototypes,
    transform_state,
    num_known_classes: int,
):
    known_prototypes = known_spectral_prototypes(
        fit_data,
        transform_state,
        num_known_classes,
    )
    transformed_candidates = apply_discovery_transform(candidate_embeddings, transform_state)
    candidate_distances, _, _ = nearest_distances(
        transformed_candidates,
        unknown_prototypes,
    )
    distance_cap = float(np.percentile(candidate_distances, 95.0)) if len(candidate_distances) else float("inf")
    return {
        "known_prototypes": known_prototypes,
        "unknown_distance_cap": distance_cap,
    }


def weak_unknown_rescue_mask(
    data,
    scores,
    rejected,
    unknown_prototypes,
    transform_state,
    rescue_state,
    known_supports,
    ratio_threshold: float,
    distance_scale: float,
):
    if len(unknown_prototypes) == 0:
        return np.zeros_like(rejected, dtype=bool), np.empty(0, dtype=int)
    transformed = apply_discovery_transform(data["spectral_embedding"], transform_state)
    unknown_distance, unknown_cluster, _ = nearest_distances(transformed, unknown_prototypes)
    known_distance_matrix = (
        (transformed[:, None, :] - rescue_state["known_prototypes"][None, :, :]) ** 2
    ).sum(axis=-1)
    predictions = data["known_prediction"].astype(int)
    prediction_known_distance = known_distance_matrix[
        np.arange(len(predictions)),
        predictions,
    ]
    ratios = unknown_distance / np.maximum(prediction_known_distance, 1e-8)
    score_floor = np.asarray(
        [known_supports.get(int(prediction), -np.inf) for prediction in predictions],
        dtype=float,
    )
    distance_cap = rescue_state["unknown_distance_cap"] * distance_scale
    rescued = (
        ~rejected
        & (scores > score_floor)
        & (ratios < ratio_threshold)
        & (unknown_distance <= distance_cap)
    )
    return rescued, unknown_cluster


def evaluation_cluster_mapping(cluster_labels, true_labels, unknown_indices):
    """Hungarian mapping for evaluation only; it does not train the model."""

    from scipy.optimize import linear_sum_assignment

    clusters = sorted(np.unique(cluster_labels).tolist())
    unknown_indices = list(unknown_indices)
    matrix = np.zeros((len(clusters), len(unknown_indices)), dtype=int)
    for row, cluster in enumerate(clusters):
        for column, true_index in enumerate(unknown_indices):
            matrix[row, column] = np.sum(
                (cluster_labels == cluster) & (true_labels == true_index)
            )
    mapping = {}
    if matrix.size:
        rows, columns = linear_sum_assignment(-matrix)
        for row, column in zip(rows, columns):
            mapping[clusters[row]] = unknown_indices[column]
    for cluster in clusters:
        if cluster not in mapping:
            row = clusters.index(cluster)
            mapping[cluster] = unknown_indices[int(np.argmax(matrix[row]))]
    return mapping


def clustering_metrics(true_labels, cluster_labels):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    return {
        "ARI": float(adjusted_rand_score(true_labels, cluster_labels)),
        "NMI": float(normalized_mutual_info_score(true_labels, cluster_labels)),
    }


def confusion_matrix(true_labels, predicted_labels, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, predicted in zip(true_labels, predicted_labels):
        matrix[int(true), int(predicted)] += 1
    return matrix


def classification_metrics(matrix, known_indices, unknown_indices):
    true_positive = np.diag(matrix).astype(float)
    precision = true_positive / np.maximum(matrix.sum(axis=0), 1)
    recall = true_positive / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "specific_class_accuracy": float(true_positive.sum() / matrix.sum()),
        "macro_f1": float(f1.mean()),
        "known_macro_recall": float(recall[list(known_indices)].mean()),
        "unknown_specific_macro_recall": float(
            recall[list(unknown_indices)].mean()
        ),
    }


def plot_confusion(matrix, class_names, path):
    if skip_plots():
        return
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list(
        "paper_purple",
        ["#FFFFFF", "#F3EFF9", "#D7C7EA", "#9A6AC8", "#5B0F8B"],
    )
    figure, axis = plt.subplots(figsize=(6.2, 5.6))
    image = axis.imshow(normalized, cmap=cmap, vmin=0, vmax=1)
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            percentage = normalized[row, column] * 100.0
            count = int(matrix[row, column])
            color = "white" if normalized[row, column] >= 0.55 else "black"
            axis.text(
                column,
                row,
                f"{percentage:.1f}%\n{count}",
                ha="center",
                va="center",
                fontsize=11,
                color=color,
                family="serif",
            )
    axis.set_xticks(range(len(class_names)), class_names)
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predict label", fontsize=13, family="serif")
    axis.set_ylabel("Truth label", fontsize=13, family="serif")
    axis.tick_params(axis="x", labelrotation=45)
    for spine in axis.spines.values():
        spine.set_linewidth(1.1)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, class_names, path, seed: int):
    if skip_plots():
        return
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    if len(embeddings) < 3:
        return
    if len(embeddings) > 1200:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(embeddings), size=1200, replace=False))
        embeddings = embeddings[selected]
        labels = labels[selected]
    scaled = StandardScaler().fit_transform(embeddings)
    perplexity = min(35, max(5, (len(scaled) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    coordinates = tsne.fit_transform(scaled)
    colors = ["#1f77b4", "#009E3D", "#ff7f0e", "#d62728", "#7f3c8d"]
    figure, axis = plt.subplots(figsize=(6.2, 5.0))
    for index, class_name in enumerate(class_names):
        mask = labels == index
        if not np.any(mask):
            continue
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=18,
            alpha=0.90,
            color=colors[index % len(colors)],
            label=class_name,
            edgecolors="none",
        )
    axis.legend(frameon=True, fontsize=10)
    axis.set_xlabel("t-SNE 1", fontsize=12, family="serif")
    axis.set_ylabel("t-SNE 2", fontsize=12, family="serif")
    for spine in axis.spines.values():
        spine.set_linewidth(1.1)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def plot_training(history, path):
    if skip_plots():
        return
    figure, axis = plt.subplots(figsize=(10, 5))
    for key in ("total", "reconstruction", "classification", "contrastive", "boundary"):
        axis.plot([row[key] for row in history], label=key)
    axis.legend()
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def save_predictions(
    path,
    samples,
    class_names,
    true_labels,
    known_predictions,
    rejected,
    unknown_clusters,
    mapped_predictions,
    scores,
    confidence,
    logit_margin,
):
    cluster_pointer = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file",
                "true_label",
                "known_head_prediction",
                "rejected_as_unknown",
                "discovered_unknown_cluster",
                "evaluation_mapped_prediction",
                "novelty_score",
                "confidence",
                "classification_margin",
            ]
        )
        for index, ((sample_path, _), true_label) in enumerate(zip(samples, true_labels)):
            if rejected[index]:
                cluster = int(unknown_clusters[cluster_pointer])
                operational = f"U{cluster}"
                cluster_pointer += 1
            else:
                operational = ""
            writer.writerow(
                [
                    str(sample_path),
                    class_names[int(true_label)],
                    class_names[int(known_predictions[index])],
                    bool(rejected[index]),
                    operational,
                    class_names[int(mapped_predictions[index])],
                    f"{float(scores[index]):.10f}",
                    f"{float(confidence[index]):.10f}",
                    f"{float(logit_margin[index]):.10f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--model-file", type=Path, default=MODEL_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--unknown-classes", nargs="+", default=["F1", "F2", "F3"])
    parser.add_argument("--unknown-clusters", type=int, default=3)
    parser.add_argument("--max-unknown-clusters", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--input-length", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=192)
    parser.add_argument("--calibration-ratio", type=float, default=0.0)
    parser.add_argument("--threshold-percentile", type=float, default=97.5)
    parser.add_argument("--known-threshold-percentile", type=float, default=95.0)
    parser.add_argument("--val-threshold-cap-percentile", type=float, default=80.0)
    parser.add_argument("--prototype-rescue-ratio", type=float, default=0.90)
    parser.add_argument("--prototype-rescue-distance-scale", type=float, default=1.35)
    parser.add_argument("--discovery-margin", type=float, default=0.0)
    parser.add_argument("--max-train-per-class", type=int, default=5)
    parser.add_argument("--max-discovery-per-class", type=int, default=None)
    parser.add_argument("--max-test-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    unknown_classes = list(dict.fromkeys(args.unknown_classes))
    invalid = set(unknown_classes) - set(ALL_AVAILABLE_CLASSES)
    if invalid:
        raise ValueError(f"Unknown dataset class names: {sorted(invalid)}")
    known_classes = [
        class_name
        for class_name in ALL_AVAILABLE_CLASSES
        if class_name not in unknown_classes
    ]
    if len(known_classes) < 2:
        raise ValueError("At least two known classes are required.")

    set_seed(args.seed)
    configure_runtime(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(args.output_dir, args)
    fit_samples, calibration_samples = split_known_train(
        args.dataset_root,
        known_classes,
        args.calibration_ratio,
        args.seed,
        args.max_train_per_class,
    )
    discovery_samples = pool_samples(
        args.dataset_root,
        "val",
        ALL_AVAILABLE_CLASSES,
        args.max_discovery_per_class,
        args.seed,
    )
    test_samples = pool_samples(
        args.dataset_root,
        "test",
        ALL_AVAILABLE_CLASSES,
        args.max_test_per_class,
        args.seed + 1000,
    )

    known_map = {name: index for index, name in enumerate(known_classes)}
    write_sample_manifest(args.output_dir, args.dataset_root, train=fit_samples,
                          calibration=calibration_samples, val=discovery_samples, test=test_samples)
    all_map = {name: index for index, name in enumerate(ALL_AVAILABLE_CLASSES)}
    known_global_indices = [all_map[name] for name in known_classes]
    unknown_global_indices = [all_map[name] for name in unknown_classes]

    model_module = load_model_module(args.model_file)
    config = model_module.DPCOpenNetConfig(
        input_length=args.input_length,
        num_known_classes=len(known_classes),
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=str(resolve_device(args.device)),
    )
    device = torch.device(config.device)
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU"
    )
    print(f"Using device: {device} ({device_name})")
    model = model_module.DPCOpenNet(config).to(device)
    main_optimizer, discriminator_optimizer = model_module.make_optimizers(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        main_optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.lr * 0.05,
    )

    fit_dataset = FaultDataset(fit_samples, known_map, args.input_length)
    calibration_dataset = FaultDataset(
        calibration_samples,
        known_map,
        args.input_length,
    )
    discovery_dataset = FaultDataset(
        discovery_samples,
        all_map,
        args.input_length,
    )
    test_dataset = FaultDataset(test_samples, all_map, args.input_length)
    fit_loader = DataLoader(fit_dataset, batch_size=args.batch_size, shuffle=True)
    fit_eval_loader = DataLoader(
        fit_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    discovery_loader = DataLoader(
        discovery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    history = []
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(
            model,
            fit_loader,
            main_optimizer,
            discriminator_optimizer,
            device,
        )
        scheduler.step()
        history.append(losses)
        if losses["total"] < best_loss:
            best_loss = losses["total"]
            torch.save(
                {"state_dict": model.state_dict(), "config": asdict(config)},
                args.output_dir / "best_model.pt",
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d}/{args.epochs}: "
                + ", ".join(f"{key}={value:.6f}" for key, value in losses.items())
            )

    means, variances = known_statistics(
        model,
        fit_eval_loader,
        device,
        len(known_classes),
    )
    fit_data = collect(model, fit_eval_loader, device, means, variances)
    normalizer = fit_normalizer(fit_data)
    discovery_data = collect(
        model,
        discovery_loader,
        device,
        means,
        variances,
    )
    fit_scores = novelty_score(fit_data, normalizer)
    discovery_scores = novelty_score(discovery_data, normalizer)
    (
        thresholds,
        confidence_thresholds,
        margin_thresholds,
        threshold_details,
    ) = adaptive_fewshot_thresholds(
        fit_scores,
        discovery_scores,
        fit_data,
        discovery_data,
        len(known_classes),
        args.known_threshold_percentile,
        args.val_threshold_cap_percentile,
    )
    discovery_rejected = reject_mask(
        discovery_scores,
        discovery_data["known_prediction"],
        thresholds,
        args.discovery_margin,
        discovery_data["confidence"],
        confidence_thresholds,
        discovery_data["margin"],
        margin_thresholds,
    )
    candidate_embeddings = discovery_data["spectral_embedding"][
        discovery_rejected
    ]
    candidate_true_labels = discovery_data["true_label"][discovery_rejected]
    if len(candidate_embeddings) == 0:
        # Conservative fallback: use the most novel 5% of the discovery pool.
        count = max(1, int(round(len(discovery_scores) * 0.05)))
        selected = np.argsort(discovery_scores)[-count:]
        discovery_rejected[selected] = True
        candidate_embeddings = discovery_data["spectral_embedding"][selected]
        candidate_true_labels = discovery_data["true_label"][selected]

    known_supports = {
        int(index): float(details["known_support"])
        for index, details in threshold_details.get("class_details", {}).items()
    }
    if not known_supports:
        known_supports = {
            int(index): float(threshold)
            for index, threshold in thresholds.items()
        }

    (
        preliminary_cluster_labels,
        preliminary_unknown_prototypes,
        _,
        preliminary_discovery_transform,
    ) = discover_unknown_prototypes(
        candidate_embeddings,
        args.unknown_clusters,
        args.max_unknown_clusters,
        args.seed,
    )
    rescue_state = prototype_rescue_state(
        fit_data,
        candidate_embeddings,
        preliminary_unknown_prototypes,
        preliminary_discovery_transform,
        len(known_classes),
    )
    discovery_rescued, _ = weak_unknown_rescue_mask(
        discovery_data,
        discovery_scores,
        discovery_rejected,
        preliminary_unknown_prototypes,
        preliminary_discovery_transform,
        rescue_state,
        known_supports,
        args.prototype_rescue_ratio,
        args.prototype_rescue_distance_scale,
    )
    if np.any(discovery_rescued):
        discovery_rejected = discovery_rejected | discovery_rescued
        candidate_embeddings = discovery_data["spectral_embedding"][discovery_rejected]
        candidate_true_labels = discovery_data["true_label"][discovery_rejected]

    (
        discovery_cluster_labels,
        unknown_prototypes,
        silhouette_scores,
        discovery_transform,
    ) = discover_unknown_prototypes(
        candidate_embeddings,
        args.unknown_clusters,
        args.max_unknown_clusters,
        args.seed,
    )
    rescue_state = prototype_rescue_state(
        fit_data,
        candidate_embeddings,
        unknown_prototypes,
        discovery_transform,
        len(known_classes),
    )
    cluster_mapping = evaluation_cluster_mapping(
        discovery_cluster_labels,
        candidate_true_labels,
        unknown_global_indices,
    )

    test_data = collect(model, test_loader, device, means, variances)
    test_scores = novelty_score(test_data, normalizer)
    test_rejected = reject_mask(
        test_scores,
        test_data["known_prediction"],
        thresholds,
        confidence=test_data["confidence"],
        confidence_thresholds=confidence_thresholds,
        logit_margin=test_data["margin"],
        margin_thresholds=margin_thresholds,
    )
    test_rescued, _ = weak_unknown_rescue_mask(
        test_data,
        test_scores,
        test_rejected,
        unknown_prototypes,
        discovery_transform,
        rescue_state,
        known_supports,
        args.prototype_rescue_ratio,
        args.prototype_rescue_distance_scale,
    )
    test_rejected = test_rejected | test_rescued
    test_unknown_embeddings = test_data["spectral_embedding"][test_rejected]
    test_unknown_clusters = (
        assign_unknown_clusters(
            test_unknown_embeddings,
            unknown_prototypes,
            discovery_transform,
        )
        if len(test_unknown_embeddings)
        else np.empty(0, dtype=int)
    )

    mapped_predictions = np.empty(len(test_scores), dtype=int)
    mapped_predictions[~test_rejected] = [
        all_map[known_classes[int(index)]]
        for index in test_data["known_prediction"][~test_rejected]
    ]
    mapped_predictions[test_rejected] = [
        cluster_mapping[int(cluster)]
        for cluster in test_unknown_clusters
    ]

    matrix = confusion_matrix(
        test_data["true_label"],
        mapped_predictions,
        len(ALL_AVAILABLE_CLASSES),
    )
    final_metrics = classification_metrics(
        matrix,
        known_global_indices,
        unknown_global_indices,
    )
    rejected_true_labels = test_data["true_label"][test_rejected]
    rejected_true_unknown_mask = np.isin(
        rejected_true_labels,
        unknown_global_indices,
    )
    clustering_result = (
        clustering_metrics(
            rejected_true_labels[rejected_true_unknown_mask],
            test_unknown_clusters[rejected_true_unknown_mask],
        )
        if np.sum(rejected_true_unknown_mask) > 1
        else {"ARI": 0.0, "NMI": 0.0}
    )

    plot_training(history, args.output_dir / "training_curves.png")
    plot_confusion(
        matrix,
        ALL_AVAILABLE_CLASSES,
        args.output_dir / "specific_fault_confusion_matrix.png",
    )
    plot_tsne(
        test_data["discovery_embedding"],
        test_data["true_label"],
        ALL_AVAILABLE_CLASSES,
        args.output_dir / "tsne_visualization.png",
        args.seed,
    )
    save_predictions(
        args.output_dir / "predictions.csv",
        test_samples,
        ALL_AVAILABLE_CLASSES,
        test_data["true_label"],
        np.asarray(
            [
                all_map[known_classes[int(index)]]
                for index in test_data["known_prediction"]
            ]
        ),
        test_rejected,
        test_unknown_clusters,
        mapped_predictions,
        test_scores,
        test_data["confidence"],
        test_data["margin"],
    )

    summary = {
        'model': 'DPC-OpenNet',
        'implementation_class': 'DPCOpenNet',
        'four_modules': ['multiscale dual-view adaptive encoding', 'prototype-contrastive known-fault learning', 'reconstruction and dual-space adversarial regularization', 'residual and spectral representations'],
        'known_classes': known_classes,
        'unknown_classes_for_evaluation': unknown_classes,
        'epochs': args.epochs,
        'fit_samples': counts(fit_samples, ALL_AVAILABLE_CLASSES),
        'calibration_samples': counts(calibration_samples, ALL_AVAILABLE_CLASSES),
        'discovery_samples': counts(discovery_samples, ALL_AVAILABLE_CLASSES),
        'test_samples': counts(test_samples, ALL_AVAILABLE_CLASSES),
        'discovered_cluster_count': int(len(unknown_prototypes)),
        'discovery_feature': 'robust-whitened 64-point log spectrum',
        'silhouette_scores': silhouette_scores,
        'operational_cluster_names': [f'U{index}' for index in range(len(unknown_prototypes))],
        'evaluation_cluster_mapping': {f'U{cluster}': ALL_AVAILABLE_CLASSES[true_index] for (cluster, true_index) in cluster_mapping.items()},
        'thresholds': {known_classes[index]: threshold for (index, threshold) in thresholds.items()},
        'confidence_thresholds': {known_classes[index]: threshold for (index, threshold) in confidence_thresholds.items()},
        'margin_thresholds': {known_classes[index]: threshold for (index, threshold) in margin_thresholds.items()},
        'threshold_calibration': threshold_details,
        'prototype_rescue': {'ratio_threshold': args.prototype_rescue_ratio, 'distance_scale': args.prototype_rescue_distance_scale, 'discovery_rescued_count': int(np.sum(discovery_rescued)), 'test_rescued_count': int(np.sum(test_rescued)), 'unknown_distance_cap': rescue_state['unknown_distance_cap']},
        'metrics': final_metrics,
        'unknown_clustering_metrics': clustering_result,
        'confusion_matrix': matrix.tolist(),
        'config': asdict(config),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "known_means": means,
            "release_schema": 1,
            "known_classes": known_classes,
            "inference_mode": "multi",
            "prototype_rescue_state": rescue_state,
            "known_supports": known_supports,
            "prototype_rescue_ratio": args.prototype_rescue_ratio,
            "prototype_rescue_distance_scale": args.prototype_rescue_distance_scale,
            "known_variances": variances,
            "score_normalizer": normalizer,
            "thresholds": thresholds,
            "confidence_thresholds": confidence_thresholds,
            "margin_thresholds": margin_thresholds,
            "unknown_prototypes": unknown_prototypes,
            "discovery_transform": discovery_transform,
            "cluster_mapping_evaluation_only": cluster_mapping,
        },
        args.output_dir / "final_model.pt",
    )

    print(
        "Discovered clusters:",
        len(unknown_prototypes),
    )
    print(
        "Operational labels:",
        [f"U{i}" for i in range(len(unknown_prototypes))],
    )
    print(
        "Evaluation mapping:",
        {
            f"U{cluster}": ALL_AVAILABLE_CLASSES[index]
            for cluster, index in cluster_mapping.items()
        },
    )
    print("Final metrics:", final_metrics)
    print("Unknown clustering:", clustering_result)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

