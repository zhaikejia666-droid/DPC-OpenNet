"""DPC-OpenNet components and network defined in Sections 3.2.1--3.2.4."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class DPCOpenNetConfig:
    input_length: int = 256
    num_known_classes: int = 4
    scales: Tuple[int, ...] = (3, 4, 5, 6, 7)
    wavelet_paths: int = 6
    wavelet_kernel_size: int = 65
    latent_dim: int = 192
    projection_dim: int = 96
    residual_dim: int = 32
    spectral_points: int = 64
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    recon_weight: float = 1.0
    cls_weight: float = 1.0
    contrast_weight: float = 0.12
    prototype_weight: float = 0.08
    adversarial_weight: float = 0.01
    label_smoothing: float = 0.05
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class FixedMorletWaveletBank(nn.Module):
    """Fixed complex Morlet filters."""

    def __init__(self, scale: int, paths: int, kernel_size: int, second_order: bool=False) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        kernels = self._make_kernels(scale, paths, kernel_size, second_order)
        self.register_buffer('real', kernels[:, 0:1])
        self.register_buffer('imag', kernels[:, 1:2])

    @staticmethod
    def _make_kernels(scale: int, paths: int, kernel_size: int, second_order: bool) -> Tensor:
        t = torch.linspace(-1.0, 1.0, kernel_size)
        kernels: List[Tensor] = []
        for index in range(paths):
            octave = (index + 0.5) * max(scale, 1) / paths
            if second_order:
                octave = min(float(scale), octave + 0.75)
            dilation = 2.0 ** octave
            sigma = min(0.85, 0.1 * dilation)
            omega = 2.4 * math.pi / dilation
            envelope = torch.exp(-t.square() / (2.0 * sigma ** 2))
            real = envelope * torch.cos(omega * t * kernel_size / 8.0)
            imag = envelope * torch.sin(omega * t * kernel_size / 8.0)
            real = real - real.mean()
            imag = imag - imag.mean()
            norm = torch.sqrt(real.square().sum() + imag.square().sum()).clamp_min(1e-08)
            kernels.append(torch.stack((real / norm, imag / norm)))
        return torch.stack(kernels)

    def modulus(self, x: Tensor) -> Tensor:
        padding = self.kernel_size // 2
        real = F.conv1d(x, self.real, padding=padding)
        imag = F.conv1d(x, self.imag, padding=padding)
        return torch.sqrt(real.square() + imag.square() + 1e-08)


class FixedWaveletScattering(nn.Module):
    """Fixed first- and second-order wavelet-scattering responses."""

    def __init__(self, scale: int, paths: int, input_length: int, kernel_size: int) -> None:
        super().__init__()
        self.scale = scale
        self.paths = paths
        self.input_length = input_length
        self.first = FixedMorletWaveletBank(scale, paths, kernel_size, second_order=False)
        self.second = FixedMorletWaveletBank(scale, paths, kernel_size, second_order=True)

    def forward(self, x: Tensor) -> Tensor:
        first = self.first.modulus(x)
        padding = self.second.kernel_size // 2
        real = F.conv1d(first, self.second.real, padding=padding, groups=self.paths)
        imag = F.conv1d(first, self.second.imag, padding=padding, groups=self.paths)
        second = torch.sqrt(real.square() + imag.square() + 1e-08)
        pool = min(2 ** self.scale, self.input_length)
        coeff = F.avg_pool1d(second, kernel_size=pool, stride=pool)
        view = coeff.flatten(1).unsqueeze(1)
        view = F.interpolate(view, size=self.input_length, mode='linear', align_corners=False)
        return view


class EfficientChannelAttention(nn.Module):
    """Local channel interactions in a branch feature map."""

    def __init__(self, channels: int, kernel_size: int=3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        weights = x.mean(dim=-1, keepdim=False).unsqueeze(1)
        weights = torch.sigmoid(self.conv(weights)).squeeze(1).unsqueeze(-1)
        return x * weights


class DepthwiseResidualBlock(nn.Module):
    """Depthwise residual convolution with efficient channel attention."""

    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels, bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise = nn.Sequential(nn.Conv1d(channels, channels * 2, 1), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(channels * 2, channels, 1))
        self.eca = EfficientChannelAttention(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pointwise(x)
        x = self.eca(x)
        return F.gelu(x + residual)


class AttentiveStatisticsPooling(nn.Module):
    """Concatenate attentive mean, standard deviation and temporal maximum."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(nn.Conv1d(channels, channels // 2, 1), nn.Tanh(), nn.Conv1d(channels // 2, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        weights = torch.softmax(self.attention(x), dim=-1)
        mean = torch.sum(weights * x, dim=-1)
        variance = torch.sum(weights * (x - mean.unsqueeze(-1)).square(), dim=-1)
        std = torch.sqrt(variance.clamp_min(1e-06))
        max_value = x.max(dim=-1).values
        return torch.cat((mean, std, max_value), dim=1)


class DualViewEncodingBranch(nn.Module):
    """Gate raw-signal and wavelet-scattering views at one scale."""

    def __init__(self, scale: int, config: DPCOpenNetConfig, kernel_size: int) -> None:
        super().__init__()
        self.scattering = FixedWaveletScattering(scale, config.wavelet_paths, config.input_length, config.wavelet_kernel_size)
        self.view_gate = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 2))
        self.stem = nn.Sequential(nn.Conv1d(2, 32, kernel_size, stride=2, padding=kernel_size // 2, bias=False), nn.BatchNorm1d(32), nn.GELU(), DepthwiseResidualBlock(32, 7, config.dropout), nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(64), nn.GELU(), DepthwiseResidualBlock(64, 7, config.dropout), nn.Conv1d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(128), nn.GELU(), DepthwiseResidualBlock(128, 5, config.dropout))
        self.pool = AttentiveStatisticsPooling(128)
        self.to_latent = nn.Sequential(nn.Linear(128 * 3, config.latent_dim), nn.LayerNorm(config.latent_dim), nn.GELU(), nn.Dropout(config.dropout))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        scattering = self.scattering(x)
        views = torch.cat((x, scattering), dim=1)
        gate = torch.softmax(self.view_gate(views), dim=-1)
        gated_views = views * gate.unsqueeze(-1)
        features = self.stem(gated_views)
        latent = self.to_latent(self.pool(features))
        return (latent, gate)


class ResidualScaleAggregation(nn.Module):
    """Attention-weighted branch aggregation with an unweighted residual path."""

    def __init__(self, latent_dim: int, num_scales: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(latent_dim, latent_dim // 2), nn.Tanh(), nn.Linear(latent_dim // 2, 1))
        self.norm = nn.LayerNorm(latent_dim)
        self.num_scales = num_scales

    def forward(self, latents: List[Tensor]) -> Tuple[Tensor, Tensor]:
        stacked = torch.stack(latents, dim=1)
        weights = torch.softmax(self.score(stacked).squeeze(-1), dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        fused = self.norm(fused + stacked.mean(dim=1))
        return (fused, weights)


class CosinePrototypeClassifier(nn.Module):
    """Known-class logits from normalized embeddings and learnable prototypes."""

    def __init__(self, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, latent_dim))
        self.log_scale = nn.Parameter(torch.tensor(math.log(12.0)))

    def forward(self, z: Tensor) -> Tensor:
        z = F.normalize(z, dim=1)
        weight = F.normalize(self.weight, dim=1)
        scale = self.log_scale.exp().clamp(1.0, 50.0)
        return scale * F.linear(z, weight)

    def prototypes(self) -> Tensor:
        return F.normalize(self.weight, dim=1)


class TransposedConvolutionStage(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, upsample: bool=True) -> None:
        super().__init__()
        self.tconv = nn.Sequential(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2), nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True))
        self.upsample = upsample

    def forward(self, x: Tensor) -> Tensor:
        x = self.tconv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        return x


class ReconstructionBranch(nn.Module):

    def __init__(self, final_kernel: int, config: DPCOpenNetConfig) -> None:
        super().__init__()
        self.input_length = config.input_length
        self.unflatten_channels = 64
        self.unflatten_length = config.latent_dim // self.unflatten_channels
        self.deconv1 = TransposedConvolutionStage(64, 64, kernel_size=3, upsample=True)
        self.deconv2 = TransposedConvolutionStage(64, 32, kernel_size=3, upsample=True)
        self.final = nn.ConvTranspose1d(32, 1, kernel_size=final_kernel, stride=16)

    def forward(self, z: Tensor) -> Tensor:
        x = z.view(z.size(0), self.unflatten_channels, self.unflatten_length)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.final(x)
        x = F.interpolate(x, size=self.input_length, mode='linear', align_corners=False)
        return x


class MultibranchReconstructionDecoder(nn.Module):
    """Average the reconstructions of five transposed-convolution branches."""

    def __init__(self, config: DPCOpenNetConfig) -> None:
        super().__init__()
        final_kernels = (256, 128, 64, 32, 16)
        self.branches = nn.ModuleList((ReconstructionBranch(k, config) for k in final_kernels))

    def forward(self, z: Tensor, return_branch_recons: bool=False) -> Tensor | Tuple[Tensor, List[Tensor]]:
        branch_recons = [branch(z) for branch in self.branches]
        fused = torch.stack(branch_recons, dim=0).mean(dim=0)
        if return_branch_recons:
            return (fused, branch_recons)
        return fused


class AdversarialDiscriminator(nn.Module):

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(inplace=True), nn.Linear(512, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.flatten(1))


class ResidualMorphologyEncoder(nn.Module):
    """Normalized 32-dimensional descriptor of absolute residual morphology."""

    def __init__(self, residual_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, 16, 9, stride=2, padding=4, bias=False), nn.BatchNorm1d(16), nn.GELU(), nn.Conv1d(16, 32, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(32), nn.GELU(), nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(64), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, residual_dim))

    def forward(self, residual: Tensor) -> Tensor:
        return F.normalize(self.net(residual), dim=1)


class FixedSpectralDescriptor(nn.Module):
    """Non-DC log-amplitude spectrum after resampling to 64 points."""

    def __init__(self, spectral_points: int=64) -> None:
        super().__init__()
        if spectral_points < 8 or spectral_points % 2:
            raise ValueError('spectral_points must be an even integer >= 8.')
        self.spectral_points = spectral_points

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        compact = F.interpolate(x, size=self.spectral_points, mode='linear', align_corners=True)
        spectrum = torch.fft.rfft(compact.squeeze(1), dim=-1).abs()
        return torch.log1p(spectrum[:, 1:])


class LatentSpaceDiscriminator(AdversarialDiscriminator):
    """Discriminate encoded latents from uniformly sampled latent codes."""


class SignalSpaceDiscriminator(AdversarialDiscriminator):
    """Discriminate reconstructions from decoded random latent codes."""


class MultiscaleDualViewAdaptiveEncoding(nn.Module):
    """Section 3.2.1: Multiscale dual-view adaptive encoding."""

    def __init__(self, config: DPCOpenNetConfig) -> None:
        super().__init__()
        kernels = (9, 15, 21, 31, 41)
        self.dual_view_branches = nn.ModuleList(
            DualViewEncodingBranch(scale, config, kernel)
            for scale, kernel in zip(config.scales, kernels)
        )
        self.residual_scale_aggregation = ResidualScaleAggregation(config.latent_dim, len(config.scales))
        self.latent_projection = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.Tanh(),
        )

    def forward(self, x: Tensor):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        branch_latents = []
        view_gates = []
        for branch in self.dual_view_branches:
            latent, gate = branch(x)
            branch_latents.append(latent)
            view_gates.append(gate)
        fused, scale_weights = self.residual_scale_aggregation(branch_latents)
        return {
            "z": self.latent_projection(fused),
            "branch_latents": branch_latents,
            "scale_weights": scale_weights,
            "view_gates": torch.stack(view_gates, dim=1),
        }


class PrototypeContrastiveKnownFaultLearning(nn.Module):
    """Section 3.2.2: Prototype-contrastive known-fault learning."""

    def __init__(self, config: DPCOpenNetConfig) -> None:
        super().__init__()
        self.semantic_projection = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim, config.projection_dim),
        )
        self.cosine_prototype_classifier = CosinePrototypeClassifier(config.latent_dim, config.num_known_classes)

    def project(self, z: Tensor) -> Tensor:
        return F.normalize(self.semantic_projection(z), dim=1)

    def forward(self, z: Tensor) -> Tensor:
        return self.cosine_prototype_classifier(z)


class ReconstructionAndDualSpaceAdversarialRegularization(nn.Module):
    """Section 3.2.3: Reconstruction and dual-space adversarial regularization."""

    def __init__(self, config: DPCOpenNetConfig) -> None:
        super().__init__()
        self.multibranch_reconstruction_decoder = MultibranchReconstructionDecoder(config)
        self.latent_space_discriminator = LatentSpaceDiscriminator(config.latent_dim)
        self.signal_space_discriminator = SignalSpaceDiscriminator(config.input_length)

    def forward(self, z: Tensor):
        return self.multibranch_reconstruction_decoder(z, return_branch_recons=True)


class ResidualAndSpectralRepresentations(nn.Module):
    """Section 3.2.4: Residual and spectral representations."""

    def __init__(self, config: DPCOpenNetConfig) -> None:
        super().__init__()
        self.residual_morphology_encoder = ResidualMorphologyEncoder(config.residual_dim)
        self.fixed_spectral_descriptor = FixedSpectralDescriptor(config.spectral_points)

    def forward(self, x: Tensor, recon: Tensor, projection: Tensor, spectral_embedding=None):
        residual = torch.abs(x - recon)
        residual_embedding = self.residual_morphology_encoder(residual)
        if spectral_embedding is None:
            spectral_embedding = self.fixed_spectral_descriptor(x)
        discovery_embedding = F.normalize(torch.cat((projection, residual_embedding), dim=1), dim=1)
        return {"residual": residual, "residual_embedding": residual_embedding,
                "spectral_embedding": spectral_embedding, "discovery_embedding": discovery_embedding}


class DPCOpenNet(nn.Module):
    """Dual-view Prototype-Contrastive Network for Open-world Fault Diagnosis."""

    _CHECKPOINT_PREFIXES = {
        "scale_encoders": "adaptive_encoding.dual_view_branches",
        "scale_fusion": "adaptive_encoding.residual_scale_aggregation",
        "latent_head": "adaptive_encoding.latent_projection",
        "projector": "known_fault_learning.semantic_projection",
        "classifier": "known_fault_learning.cosine_prototype_classifier",
        "decoder": "reconstruction_regularization.multibranch_reconstruction_decoder",
        "encoder_discriminator": "reconstruction_regularization.latent_space_discriminator",
        "decoder_discriminator": "reconstruction_regularization.signal_space_discriminator",
        "residual_encoder": "residual_spectral_representations.residual_morphology_encoder",
        "spectral_topology": "residual_spectral_representations.fixed_spectral_descriptor",
    }

    def __init__(self, config: DPCOpenNetConfig) -> None:
        super().__init__()
        self.config = config
        self.adaptive_encoding = MultiscaleDualViewAdaptiveEncoding(config)
        self.known_fault_learning = PrototypeContrastiveKnownFaultLearning(config)
        self.reconstruction_regularization = ReconstructionAndDualSpaceAdversarialRegularization(config)
        self.residual_spectral_representations = ResidualAndSpectralRepresentations(config)

    def encode(self, x: Tensor):
        encoded = self.adaptive_encoding(x)
        encoded["projection"] = self.known_fault_learning.project(encoded["z"])
        return encoded

    def decode(self, z: Tensor):
        return self.reconstruction_regularization(z)

    def forward(self, x: Tensor, spectral_embedding: Tensor | None = None):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        encoded = self.encode(x)
        recon, branch_recons = self.decode(encoded["z"])
        logits = self.known_fault_learning(encoded["z"])
        representations = self.residual_spectral_representations(x, recon, encoded["projection"], spectral_embedding)
        return {**encoded, "recon": recon, "branch_recons": branch_recons, "logits": logits, **representations}

    @torch.no_grad()
    def infer(self, x: Tensor, spectral_embedding: Tensor | None = None):
        return self.forward(x, spectral_embedding)

    @staticmethod
    def energy(logits: Tensor) -> Tensor:
        return -torch.logsumexp(logits, dim=1)

    def known_prototypes(self) -> Tensor:
        return self.known_fault_learning.cosine_prototype_classifier.prototypes()

    @property
    def classifier(self):
        return self.known_fault_learning.cosine_prototype_classifier

    @property
    def encoder_discriminator(self):
        return self.reconstruction_regularization.latent_space_discriminator

    @property
    def decoder_discriminator(self):
        return self.reconstruction_regularization.signal_space_discriminator

    @classmethod
    def checkpoint_key(cls, key):
        for old, new in cls._CHECKPOINT_PREFIXES.items():
            if key == old or key.startswith(old + "."):
                return new + key[len(old):]
        return key

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Accept state-dict checkpoints saved before component names were aligned.
        renamed = OrderedDict((self.checkpoint_key(key), value) for key, value in state_dict.items())
        if hasattr(state_dict, "_metadata"):
            renamed._metadata = {self.checkpoint_key(key): value for key, value in state_dict._metadata.items()}
        return super().load_state_dict(renamed, strict=strict, **kwargs)


def make_optimizers(model: DPCOpenNet):
    discriminator_ids = {id(parameter) for module in (model.encoder_discriminator, model.decoder_discriminator) for parameter in module.parameters()}
    main_parameters: Iterable[nn.Parameter] = (parameter for parameter in model.parameters() if id(parameter) not in discriminator_ids)
    discriminator_parameters = list(model.encoder_discriminator.parameters()) + list(model.decoder_discriminator.parameters())
    main_optimizer = torch.optim.AdamW(main_parameters, lr=model.config.learning_rate, weight_decay=model.config.weight_decay)
    discriminator_optimizer = torch.optim.AdamW(discriminator_parameters, lr=model.config.learning_rate, weight_decay=model.config.weight_decay)
    return (main_optimizer, discriminator_optimizer)
