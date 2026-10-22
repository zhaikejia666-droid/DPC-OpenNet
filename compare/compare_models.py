from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class CompareConfig:
    input_length: int = 256
    num_classes: int = 4
    latent_dim: int = 128
    projection_dim: int = 64
    dropout: float = 0.10


def _as_signal(x: Tensor) -> Tensor:
    if x.ndim == 2:
        x = x.unsqueeze(1)
    return x


class ConvBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(channels_out),
            nn.GELU(),
            nn.Conv1d(channels_out, channels_out, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels_out),
        )
        self.skip = (
            nn.Identity()
            if channels_in == channels_out and stride == 1
            else nn.Conv1d(channels_in, channels_out, 1, stride=stride, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.net(x) + self.skip(x))


class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim: int, input_length: int):
        super().__init__()
        self.input_length = input_length
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, input_length),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z).unsqueeze(1)


class CompareBase(nn.Module):
    method_name = "BASE"

    def __init__(self, config: CompareConfig):
        super().__init__()
        self.config = config

    def _finish(self, z: Tensor, logits: Tensor, reconstruction: Tensor, extra: Dict[str, Tensor] | None = None):
        projection = F.normalize(z, dim=1)
        result: Dict[str, Tensor] = {
            "z": z,
            "embedding": projection,
            "projection": projection,
            "logits": logits,
            "reconstruction": reconstruction,
        }
        if extra:
            result.update(extra)
        return result


class CRFDE(CompareBase):
    """Convolutional residual feature distribution encoder baseline."""

    method_name = "CRFDE"

    def __init__(self, config: CompareConfig):
        super().__init__(config)
        self.encoder = nn.Sequential(
            ConvBlock(1, 32, 7, 2),
            ConvBlock(32, 64, 5, 2),
            ConvBlock(64, 96, 5, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.mu = nn.Linear(96, config.latent_dim)
        self.logvar = nn.Linear(96, config.latent_dim)
        self.classifier = nn.Linear(config.latent_dim, config.num_classes)
        self.decoder = SimpleDecoder(config.latent_dim, config.input_length)

    def forward(self, x: Tensor):
        x = _as_signal(x)
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-5.0, 3.0)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        logits = self.classifier(z)
        recon = self.decoder(z)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return self._finish(z, logits, recon, {"kl_loss": kl})


class DVPN(CompareBase):
    """Deep variational prototype network baseline."""

    method_name = "DVPN"

    def __init__(self, config: CompareConfig):
        super().__init__(config)
        self.encoder = nn.Sequential(
            ConvBlock(1, 32, 9, 2),
            ConvBlock(32, 64, 7, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.mu = nn.Linear(64, config.latent_dim)
        self.logvar = nn.Linear(64, config.latent_dim)
        self.prototypes = nn.Parameter(torch.randn(config.num_classes, config.latent_dim) * 0.02)
        self.scale = nn.Parameter(torch.tensor(12.0))
        self.decoder = SimpleDecoder(config.latent_dim, config.input_length)

    def forward(self, x: Tensor):
        x = _as_signal(x)
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-5.0, 3.0)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        logits = self.scale.clamp(1.0, 30.0) * F.normalize(z, dim=1) @ F.normalize(self.prototypes, dim=1).t()
        recon = self.decoder(z)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return self._finish(z, logits, recon, {"kl_loss": kl})


class GEBS(CompareBase):
    """Gaussian embedding boundary scoring baseline."""

    method_name = "GEBS"

    def __init__(self, config: CompareConfig):
        super().__init__(config)
        self.encoder = nn.Sequential(
            ConvBlock(1, 32, 5, 2),
            ConvBlock(32, 64, 5, 2),
            ConvBlock(64, 96, 3, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(96, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
        )
        self.centers = nn.Parameter(torch.randn(config.num_classes, config.latent_dim) * 0.05)
        self.log_sigma = nn.Parameter(torch.zeros(config.num_classes, config.latent_dim))
        self.decoder = SimpleDecoder(config.latent_dim, config.input_length)

    def forward(self, x: Tensor):
        x = _as_signal(x)
        z = self.encoder(x)
        sigma = F.softplus(self.log_sigma) + 1e-4
        diff = z[:, None, :] - self.centers[None, :, :]
        logits = -0.5 * ((diff / sigma[None, :, :]).pow(2).mean(dim=2) + sigma.log().mean(dim=1)[None, :])
        recon = self.decoder(z)
        return self._finish(z, logits, recon)


class OFER(CompareBase):
    """Open feature envelope recognition baseline."""

    method_name = "OFER"

    def __init__(self, config: CompareConfig):
        super().__init__(config)
        self.encoder = nn.Sequential(
            ConvBlock(1, 24, 11, 2),
            ConvBlock(24, 48, 7, 2),
            ConvBlock(48, 96, 5, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(96, config.latent_dim),
            nn.GELU(),
        )
        self.classifier = nn.utils.parametrizations.spectral_norm(nn.Linear(config.latent_dim, config.num_classes))
        self.radius = nn.Parameter(torch.ones(config.num_classes) * 0.65)
        self.centers = nn.Parameter(torch.randn(config.num_classes, config.latent_dim) * 0.02)
        self.decoder = SimpleDecoder(config.latent_dim, config.input_length)

    def forward(self, x: Tensor):
        x = _as_signal(x)
        z = self.encoder(x)
        logits = self.classifier(z)
        distance = torch.cdist(F.normalize(z, dim=1), F.normalize(self.centers, dim=1))
        envelope_penalty = F.relu(distance.min(dim=1).values - F.softplus(self.radius).mean()).mean()
        recon = self.decoder(z)
        return self._finish(z, logits, recon, {"regularization_loss": 0.02 * envelope_penalty})


class PDAgg(CompareBase):
    """Prototype distribution aggregation baseline."""

    method_name = "PDAgg"

    def __init__(self, config: CompareConfig, prototypes_per_class: int = 3):
        super().__init__(config)
        self.encoder = nn.Sequential(
            ConvBlock(1, 32, 7, 2),
            ConvBlock(32, 64, 5, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, config.latent_dim),
            nn.GELU(),
        )
        self.prototypes = nn.Parameter(
            torch.randn(config.num_classes, prototypes_per_class, config.latent_dim) * 0.03
        )
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.decoder = SimpleDecoder(config.latent_dim, config.input_length)

    def forward(self, x: Tensor):
        x = _as_signal(x)
        z = self.encoder(x)
        zn = F.normalize(z, dim=1)
        pn = F.normalize(self.prototypes, dim=2)
        sim = torch.einsum("bd,ckd->bck", zn, pn)
        logits = self.scale.clamp(1.0, 30.0) * torch.logsumexp(sim * 8.0, dim=2) / 8.0
        recon = self.decoder(z)
        return self._finish(z, logits, recon)


class SSO_FC(CompareBase):
    """Self-supervised orthogonal feature clustering baseline."""

    method_name = "SSO-FC"

    def __init__(self, config: CompareConfig):
        super().__init__(config)
        self.encoder = nn.Sequential(
            ConvBlock(1, 32, 9, 2),
            ConvBlock(32, 64, 5, 2),
            ConvBlock(64, 96, 3, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(96, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(config.latent_dim, config.num_classes, bias=False)
        self.decoder = SimpleDecoder(config.latent_dim, config.input_length)

    def forward(self, x: Tensor):
        x = _as_signal(x)
        z = self.encoder(x)
        weight = F.normalize(self.classifier.weight, dim=1)
        logits = 14.0 * F.normalize(z, dim=1) @ weight.t()
        gram = weight @ weight.t()
        eye = torch.eye(gram.size(0), device=gram.device)
        ortho = (gram - eye).pow(2).mean()
        recon = self.decoder(z)
        return self._finish(z, logits, recon, {"regularization_loss": 0.05 * ortho})


class TNPNet(CompareBase):
    """Transformer neural prototype network baseline."""

    method_name = "TNPNet"

    def __init__(self, config: CompareConfig):
        super().__init__(config)
        self.patch = nn.Conv1d(1, 64, kernel_size=9, stride=4, padding=4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(nn.Linear(64, config.latent_dim), nn.GELU())
        self.prototypes = nn.Parameter(torch.randn(config.num_classes, config.latent_dim) * 0.02)
        self.decoder = SimpleDecoder(config.latent_dim, config.input_length)

    def forward(self, x: Tensor):
        x = _as_signal(x)
        tokens = self.patch(x).transpose(1, 2)
        tokens = self.transformer(tokens)
        z = self.head(tokens.mean(dim=1))
        logits = 12.0 * F.normalize(z, dim=1) @ F.normalize(self.prototypes, dim=1).t()
        recon = self.decoder(z)
        return self._finish(z, logits, recon)


class FEUG(CompareBase):
    """Feature-enhanced unknown generation model from the exemplar paper."""

    method_name = "FEUG"

    def __init__(self, config: CompareConfig):
        super().__init__(config)
        self.local = nn.Sequential(
            ConvBlock(1, 32, 5, 2),
            ConvBlock(32, 64, 3, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.global_branch = nn.Sequential(
            ConvBlock(1, 32, 15, 2),
            ConvBlock(32, 64, 9, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Linear(config.latent_dim, config.num_classes)
        self.unknown_generator = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        self.decoder = SimpleDecoder(config.latent_dim, config.input_length)

    def forward(self, x: Tensor):
        x = _as_signal(x)
        z = self.fusion(torch.cat([self.local(x), self.global_branch(x)], dim=1))
        logits = self.classifier(z)
        recon = self.decoder(z)
        extra: Dict[str, Tensor] = {}
        if self.training:
            unknown_z = z + 0.35 * torch.tanh(self.unknown_generator(z.detach()))
            unknown_logits = self.classifier(unknown_z)
            uniform = torch.full_like(unknown_logits, 1.0 / unknown_logits.size(1))
            unknown_loss = F.kl_div(
                F.log_softmax(unknown_logits, dim=1),
                uniform,
                reduction="batchmean",
            )
            separation = F.relu(0.60 - (unknown_z - z.detach()).norm(dim=1)).mean()
            extra["regularization_loss"] = 0.05 * unknown_loss + 0.02 * separation
        return self._finish(z, logits, recon, extra)


MODEL_REGISTRY = {
    "CRFDE": CRFDE,
    "DVPN": DVPN,
    "GEBS": GEBS,
    "OFER": OFER,
    "PDAgg": PDAgg,
    "SSO-FC": SSO_FC,
    "SSO_FC": SSO_FC,
    "TNPNet": TNPNet,
    "FEUG": FEUG,
}


def build_model(name: str, config: CompareConfig) -> nn.Module:
    try:
        return MODEL_REGISTRY[name](config)
    except KeyError as exc:
        raise ValueError(f"Unknown compare model: {name}. Choices: {sorted(MODEL_REGISTRY)}") from exc
