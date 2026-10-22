"""MS-DAWCAE comparison model and dual-discriminator training functions."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class MSDAWCAEConfig:
    input_length: int = 4096
    num_classes: int = 3
    encoding_scales: Tuple[int, ...] = (3, 4, 5, 6, 7)
    decoding_scales: Tuple[int, ...] = (1, 2, 3, 4, 5)
    wavelets_per_octave: int = 12
    latent_dim: int = 4096
    batch_size: int = 64
    epochs: int = 300
    learning_rate: float = 1e-05
    threshold_detection: float = 1.4
    threshold_multitask: float = 1.5
    recon_weight: float = 1.0
    cls_weight: float = 1.0
    adv_weight: float = 0.01
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class WaveletScattering1D(nn.Module):
    """A compact 1D Morlet scattering approximation.

    The paper uses wavelet scattering transforms with quality factor Q=12 and
    scale parameters 3, 4, 5, 6, and 7. This module builds a bank of fixed
    Morlet-like wavelets, applies convolution, takes the modulus response, and
    averages the filter responses back to a single 1x4096 signal.
    """

    def __init__(self, scale_parameter: int, q: int=12, input_length: int=4096, kernel_size: int=129) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.scale_parameter = scale_parameter
        self.q = q
        self.input_length = input_length
        self.kernel_size = kernel_size
        kernels = self._build_kernels(scale_parameter, q, kernel_size)
        self.register_buffer('real_kernels', kernels[:, 0:1, :])
        self.register_buffer('imag_kernels', kernels[:, 1:2, :])

    @staticmethod
    def _build_kernels(scale_parameter: int, q: int, kernel_size: int) -> Tensor:
        t = torch.linspace(-1.0, 1.0, kernel_size)
        filters: List[Tensor] = []
        base_freq = 1.5 * math.pi
        max_octave = max(1, scale_parameter)
        for octave in range(max_octave):
            for wavelet_idx in range(q):
                scale = 2.0 ** (octave + wavelet_idx / q)
                sigma = 0.18 * scale
                freq = base_freq / scale
                envelope = torch.exp(-t ** 2 / (2.0 * sigma ** 2))
                real = envelope * torch.cos(freq * t * kernel_size / 8.0)
                imag = envelope * torch.sin(freq * t * kernel_size / 8.0)
                real = real - real.mean()
                imag = imag - imag.mean()
                norm = torch.sqrt((real ** 2 + imag ** 2).sum()).clamp_min(1e-08)
                filters.append(torch.stack((real / norm, imag / norm), dim=0))
        return torch.stack(filters, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.size(1) != 1:
            raise ValueError('MS-DAWCAE expects one-channel 1D samples.')
        padding = self.kernel_size // 2
        real = F.conv1d(x, self.real_kernels, padding=padding)
        imag = F.conv1d(x, self.imag_kernels, padding=padding)
        modulus = torch.sqrt(real.square() + imag.square() + 1e-08)
        scattered = modulus.mean(dim=1, keepdim=True)
        return F.interpolate(scattered, size=self.input_length, mode='linear', align_corners=False)


class ConvModule(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, pool: bool) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.block(x))


class EncoderBranch(nn.Module):

    def __init__(self, scale: int, first_stride: int, config: MSDAWCAEConfig) -> None:
        super().__init__()
        self.scattering = WaveletScattering1D(scale_parameter=scale, q=config.wavelets_per_octave, input_length=config.input_length)
        self.conv1 = ConvModule(1, 32, kernel_size=128, stride=first_stride, pool=True)
        self.conv2 = ConvModule(32, 64, kernel_size=3, stride=1, pool=True)
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(64), nn.Tanh())
        self.to_latent = nn.AdaptiveAvgPool1d(config.latent_dim // 64)

    def forward(self, x: Tensor) -> Tensor:
        x = self.scattering(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.to_latent(x)
        return x.flatten(1)


class Encoder(nn.Module):
    """Five-scale encoder from Fig. 10, fused by averaging."""

    def __init__(self, config: MSDAWCAEConfig) -> None:
        super().__init__()
        first_strides = (20, 36, 38, 32, 24)
        self.branches = nn.ModuleList((EncoderBranch(scale, stride, config) for (scale, stride) in zip(config.encoding_scales, first_strides)))
        self.out = nn.Tanh()

    def forward(self, x: Tensor, return_branch_latents: bool=False) -> Tensor | Tuple[Tensor, List[Tensor]]:
        branch_latents = [branch(x) for branch in self.branches]
        fused = torch.stack(branch_latents, dim=0).mean(dim=0)
        fused = self.out(fused)
        if return_branch_latents:
            return (fused, branch_latents)
        return fused


class DeconvModule(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, upsample: bool=True) -> None:
        super().__init__()
        self.tconv = nn.Sequential(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2), nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True))
        self.upsample = upsample

    def forward(self, x: Tensor) -> Tensor:
        x = self.tconv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        return x


class DecoderBranch(nn.Module):

    def __init__(self, final_kernel: int, config: MSDAWCAEConfig) -> None:
        super().__init__()
        self.input_length = config.input_length
        self.unflatten_channels = 64
        self.unflatten_length = config.latent_dim // self.unflatten_channels
        self.deconv1 = DeconvModule(64, 64, kernel_size=3, upsample=True)
        self.deconv2 = DeconvModule(64, 32, kernel_size=3, upsample=True)
        self.final = nn.ConvTranspose1d(32, 1, kernel_size=final_kernel, stride=16)

    def forward(self, z: Tensor) -> Tensor:
        x = z.view(z.size(0), self.unflatten_channels, self.unflatten_length)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.final(x)
        x = F.interpolate(x, size=self.input_length, mode='linear', align_corners=False)
        return x


class Decoder(nn.Module):
    """Five-scale decoder from Fig. 10, integrated by averaging."""

    def __init__(self, config: MSDAWCAEConfig) -> None:
        super().__init__()
        final_kernels = (256, 128, 64, 32, 16)
        self.branches = nn.ModuleList((DecoderBranch(k, config) for k in final_kernels))

    def forward(self, z: Tensor, return_branch_recons: bool=False) -> Tensor | Tuple[Tensor, List[Tensor]]:
        branch_recons = [branch(z) for branch in self.branches]
        fused = torch.stack(branch_recons, dim=0).mean(dim=0)
        if return_branch_recons:
            return (fused, branch_recons)
        return fused


class MLPDiscriminator(nn.Module):

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(inplace=True), nn.Linear(512, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.flatten(1))


class Classifier(nn.Module):

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(inplace=True), nn.Linear(512, 64), nn.ReLU(inplace=True), nn.Linear(64, num_classes))

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class MSDAWCAE(nn.Module):

    def __init__(self, config: MSDAWCAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.classifier = Classifier(config.latent_dim, config.num_classes)
        self.encoder_discriminator = MLPDiscriminator(config.latent_dim)
        self.decoder_discriminator = MLPDiscriminator(config.input_length)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def reconstruct(self, x: Tensor, return_branch_recons: bool=False) -> Tensor | Tuple[Tensor, List[Tensor]]:
        z = self.encoder(x)
        return self.decoder(z, return_branch_recons=return_branch_recons)

    def forward(self, x: Tensor) -> Dict[str, Tensor | List[Tensor]]:
        z = self.encoder(x)
        (recon, branch_recons) = self.decoder(z, return_branch_recons=True)
        logits = self.classifier(z)
        return {'z': z, 'recon': recon, 'branch_recons': branch_recons, 'logits': logits}

    @torch.no_grad()
    def reconstruction_error(self, x: Tensor) -> Tensor:
        recon = self.reconstruct(x)
        return F.mse_loss(recon, _as_signal(x), reduction='none').flatten(1).mean(dim=1)

    @torch.no_grad()
    def predict(self, x: Tensor, threshold: float) -> Tuple[Tensor, Tensor, Tensor]:
        outputs = self.forward(x)
        errors = F.mse_loss(outputs['recon'], _as_signal(x), reduction='none').flatten(1).mean(dim=1)
        known_pred = torch.argmax(outputs['logits'], dim=1)
        unknown = errors >= threshold
        labels = known_pred.clone()
        labels[unknown] = -1
        return (labels, errors, outputs['logits'])


def _as_signal(x: Tensor) -> Tensor:
    return x.unsqueeze(1) if x.ndim == 2 else x


def pseudo_latent(batch_size: int, latent_dim: int, device: torch.device) -> Tensor:
    return torch.empty(batch_size, latent_dim, device=device).uniform_(-1.0, 1.0)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def train_ms_dawcae_epoch(model: MSDAWCAE, loader: DataLoader, ae_optimizer: torch.optim.Optimizer, d_optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    cfg = model.config
    totals = {'d_loss': 0.0, 'ae_loss': 0.0, 'recon_loss': 0.0, 'cls_loss': 0.0, 'adv_loss': 0.0}
    n_batches = 0
    for (x, y) in loader:
        x = _as_signal(x.to(device).float())
        y = y.to(device).long()
        batch_size = x.size(0)
        ones = torch.ones(batch_size, 1, device=device)
        zeros = torch.zeros(batch_size, 1, device=device)
        set_requires_grad(model.encoder_discriminator, True)
        set_requires_grad(model.decoder_discriminator, True)
        d_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            z_real = model.encoder(x)
            recon_real = model.decoder(z_real)
            z_fake = pseudo_latent(batch_size, cfg.latent_dim, device)
            recon_fake = model.decoder(z_fake)
        d_loss = bce(model.encoder_discriminator(z_real), ones) + bce(model.encoder_discriminator(z_fake), zeros) + bce(model.decoder_discriminator(recon_real), ones) + bce(model.decoder_discriminator(recon_fake), zeros)
        d_loss.backward()
        d_optimizer.step()
        set_requires_grad(model.encoder_discriminator, False)
        set_requires_grad(model.decoder_discriminator, False)
        ae_optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        z = outputs['z']
        recon = outputs['recon']
        branch_recons = outputs['branch_recons']
        logits = outputs['logits']
        z_fake = pseudo_latent(batch_size, cfg.latent_dim, device)
        recon_fake = model.decoder(z_fake)
        branch_loss = torch.stack([F.mse_loss(branch_recon, x) for branch_recon in branch_recons]).mean()
        recon_loss = 0.5 * F.mse_loss(recon, x) + 0.5 * branch_loss
        cls_loss = ce(logits, y)
        adv_loss = bce(model.encoder_discriminator(z), ones) + bce(model.decoder_discriminator(recon), ones) + bce(model.decoder_discriminator(recon_fake), ones)
        ae_loss = cfg.recon_weight * recon_loss + cfg.cls_weight * cls_loss + cfg.adv_weight * adv_loss
        ae_loss.backward()
        ae_optimizer.step()
        set_requires_grad(model.encoder_discriminator, True)
        set_requires_grad(model.decoder_discriminator, True)
        totals['d_loss'] += float(d_loss.detach().cpu())
        totals['ae_loss'] += float(ae_loss.detach().cpu())
        totals['recon_loss'] += float(recon_loss.detach().cpu())
        totals['cls_loss'] += float(cls_loss.detach().cpu())
        totals['adv_loss'] += float(adv_loss.detach().cpu())
        n_batches += 1
    return {name: value / max(1, n_batches) for (name, value) in totals.items()}


def make_ms_dawcae_optimizers(model: MSDAWCAE, learning_rate: float) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    ae_params: Iterable[nn.Parameter] = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.classifier.parameters())
    d_params: Iterable[nn.Parameter] = list(model.encoder_discriminator.parameters()) + list(model.decoder_discriminator.parameters())
    return (torch.optim.Adam(ae_params, lr=learning_rate), torch.optim.Adam(d_params, lr=learning_rate))
