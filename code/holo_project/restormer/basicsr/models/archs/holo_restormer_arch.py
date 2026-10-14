"""HoloRestormer v3 for digital-holographic x4 intensity SR.

The implementation keeps the upstream Restormer path available behind explicit
switches.  With all Holo switches disabled and ``scale=1``, the forward pass
is the same computation as the upstream Restormer.  The x4 mode adds an SR
reconstruction head and the physically motivated P1--P4 modules described in
``10.11/全息超分-HoloRestormer-v3-模块重设计方案.md``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from basicsr.models.archs.restormer_arch import (
    Attention,
    Downsample,
    FeedForward,
    LayerNorm,
    OverlapPatchEmbed,
    Upsample,
)


def spectral_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a real-valued frequency mask in fp32, even under AMP."""
    with torch.cuda.amp.autocast(enabled=False):
        spectrum = torch.fft.fft2(x.float(), norm="ortho")
        result = torch.fft.ifft2(spectrum * mask.float(), norm="ortho").real
    return result.to(dtype=x.dtype)


class HoloBandSplit(nn.Module):
    """Split a feature map at the physical LR Nyquist radius.

    ``pass`` and ``edge`` together are the in-band restoration branch; ``stop``
    is the out-of-band extrapolation branch.  The masks are hard partitions,
    so their sum is exactly one (up to floating point representation).
    """

    def __init__(self, scale: int = 4, edge_width: float = 0.08):
        super().__init__()
        if int(scale) < 1:
            raise ValueError("scale must be >= 1")
        self.scale = int(scale)
        self.edge_width = float(edge_width)

    def masks(self, height: int, width: int, device=None, dtype=None):
        fy = torch.fft.fftfreq(height, device=device, dtype=torch.float32)
        fx = torch.fft.fftfreq(width, device=device, dtype=torch.float32)
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        # r_hat = radius / (1/(2*scale)) = 2*scale*radius.
        r_hat = 2.0 * self.scale * radius
        edge = self.edge_width
        m_pass = (r_hat < (1.0 - edge)).to(dtype or torch.float32)
        m_edge = ((r_hat >= (1.0 - edge)) & (r_hat < (1.0 + edge))).to(
            dtype or torch.float32
        )
        m_stop = (r_hat >= (1.0 + edge)).to(dtype or torch.float32)
        return tuple(mask[None, None] for mask in (m_pass, m_edge, m_stop))

    def band_energy(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            power = torch.fft.fft2(x.float(), norm="ortho").abs().square()
            masks = self.masks(x.shape[-2], x.shape[-1], x.device, power.dtype)
            values = [
                (power * mask).mean(dim=(-2, -1)) for mask in masks
            ]
            return torch.stack(values, dim=-1).to(dtype=x.dtype)


class FSASBranch(nn.Module):
    """Full-resolution frequency/spatial branch with no token bottleneck."""

    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DirConv(nn.Module):
    """Lightweight directional depthwise filters for fringe anisotropy."""

    def __init__(self, dim: int, n_orient: int = 6):
        super().__init__()
        self.n_orient = int(n_orient)
        # Fixed orientation kernels are represented by a learned depthwise
        # mixture.  Keeping the branches full resolution avoids a frequency
        # token bottleneck.
        kernels = []
        for i in range(self.n_orient):
            angle = math.pi * i / self.n_orient
            dx, dy = round(math.cos(angle)), round(math.sin(angle))
            kernel = torch.zeros(3, 3)
            kernel[1, 1] = -1.0
            kernel[1 + dy, 1 + dx] = 1.0
            kernels.append(kernel)
        self.register_buffer("kernels", torch.stack(kernels)[:, None])
        self.weight = nn.Parameter(torch.zeros(self.n_orient))
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.weight, dim=0)
        branches = []
        for kernel, weight in zip(self.kernels, weights):
            kernel = kernel.to(dtype=x.dtype, device=x.device)
            # Grouped convolution applies the same directional filter to every
            # channel, which is both deterministic and inexpensive.
            filtered = F.conv2d(x, kernel.expand(x.shape[1], 1, 3, 3),
                                padding=1, groups=x.shape[1])
            branches.append(filtered * weight)
        return self.proj(sum(branches))


class MultiScaleDW(nn.Module):
    """Depthwise branches covering the configured speckle correlation scales."""

    def __init__(self, dim: int, dilations=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(dim, dim, 3, 1, dilation, dilation=dilation,
                      groups=dim, bias=False)
            for dilation in dilations
        ])
        self.weight = nn.Parameter(torch.zeros(len(self.branches)))
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.weight, dim=0)
        return self.proj(sum(w * branch(x)
                             for w, branch in zip(weights, self.branches)))


class DualGate(nn.Module):
    """Symmetric channel gate between MDTA and the band branch."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, attention: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([attention, band], dim=1))
        return attention * gate + band * (1.0 - gate)


class HoloAttention(Attention):
    """MDTA with an optional nonlinear value path."""

    def __init__(self, dim, num_heads, bias, use_nonlinear_v=False):
        super().__init__(dim, num_heads, bias)
        self.use_nonlinear_v = bool(use_nonlinear_v)

    def forward(self, x):
        if not self.use_nonlinear_v:
            return super().forward(x)
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out = torch.tanh(attn @ torch.tanh(v))
        out = rearrange(out, "b head c (h w) -> b (head c) h w",
                        head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class HoloGDFN(FeedForward):
    """GDFN with an optional directional value transform."""

    def __init__(self, dim, ffn_expansion_factor, bias, n_orient=6,
                 use_dirconv=False):
        super().__init__(dim, ffn_expansion_factor, bias)
        self.use_dirconv = bool(use_dirconv)
        self.dirconv = DirConv(int(dim * ffn_expansion_factor * 2), n_orient) \
            if self.use_dirconv else nn.Identity()

    def forward(self, x):
        x = self.project_in(x)
        x = self.dirconv(self.dwconv(x))
        x1, x2 = x.chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class HoloBlock(nn.Module):
    """Restormer block with optional in-band/out-of-band physical branches."""

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias,
                 LayerNorm_type, scale=4, n_orient=6,
                 use_restore_branch=False, use_extrap_branch=False,
                 use_dirconv=False, use_multiscale=False,
                 use_nonlinear_v=False, use_checkpoint=False):
        super().__init__()
        self.use_restore_branch = bool(use_restore_branch)
        self.use_extrap_branch = bool(use_extrap_branch)
        self.use_dirconv = bool(use_dirconv)
        self.use_multiscale = bool(use_multiscale)
        self.use_checkpoint = bool(use_checkpoint)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = HoloAttention(dim, num_heads, bias, use_nonlinear_v)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = HoloGDFN(dim, ffn_expansion_factor, bias, n_orient,
                            use_dirconv=use_dirconv)
        self.split = HoloBandSplit(scale=scale)
        self.branch_r = FSASBranch(dim) if self.use_restore_branch else None
        self.branch_e = FSASBranch(dim) if self.use_extrap_branch else None
        self.dirc = DirConv(dim, n_orient) if self.use_dirconv else nn.Identity()
        self.ms = MultiScaleDW(dim) if self.use_multiscale else nn.Identity()
        self.gate = DualGate(dim) if (self.use_restore_branch or self.use_extrap_branch) else None

    def forward(self, x):
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)

    def _forward_impl(self, x):
        # This branch deliberately mirrors TransformerBlock's two expressions.
        # It is used by the strict all-switches-off compatibility test.
        if not (self.use_restore_branch or self.use_extrap_branch or
                self.use_dirconv or self.use_multiscale or
                getattr(self.attn, "use_nonlinear_v", False)):
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
            return x

        masks = self.split.masks(x.shape[-2], x.shape[-1], x.device, x.dtype)
        band = None
        if self.branch_r is not None:
            band = spectral_mask(self.branch_r(self.norm1(x)), masks[0] + masks[1])
        if self.branch_e is not None:
            extrap = spectral_mask(self.branch_e(self.norm1(x)), masks[2])
            band = extrap if band is None else band + extrap
        if band is not None:
            band = self.ms(self.dirc(band))
        attention = self.attn(self.norm1(x))
        if band is not None:
            attention = self.gate(attention, band)
        x = x + attention
        x = x + self.ffn(self.norm2(x))
        return x


class HoloStem(nn.Module):
    """Learnable Box--Cox intensity-domain and homomorphic-domain stems."""

    def __init__(self, in_chans=3, embed_dim=48, use_homomorphic=True):
        super().__init__()
        self.use_homomorphic = bool(use_homomorphic)
        if self.use_homomorphic:
            self.p_raw = nn.Parameter(torch.tensor(-3.0))
            half = embed_dim // 2
            self.stem_id = nn.Conv2d(in_chans, half, 3, 1, 1, bias=False)
            self.stem_hm = nn.Conv2d(in_chans, embed_dim - half, 3, 1, 1,
                                     bias=False)
        else:
            self.proj = OverlapPatchEmbed(in_chans, embed_dim, bias=False).proj

    @property
    def p(self):
        return F.softplus(self.p_raw) + 0.05 if self.use_homomorphic else None

    def boxcox(self, x, eps=1e-3):
        p = self.p
        return (x.clamp_min(eps).pow(p) - 1.0) / p

    def forward(self, x):
        if not self.use_homomorphic:
            return self.proj(x)
        return torch.cat([self.stem_id(x), self.stem_hm(self.boxcox(x))], dim=1)


def _gaussian_kernel(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(2.0 * float(sigma))))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords.square()) / (2.0 * float(sigma) ** 2))
    kernel = kernel / kernel.sum()
    return kernel[:, None] * kernel[None, :]


def _lanczos_weights(in_size: int, out_size: int, device,
                     dtype, radius: int = 3) -> torch.Tensor:
    """Return center-aligned, separable Lanczos resize weights."""
    if in_size < 1 or out_size < 1:
        raise ValueError("Lanczos dimensions must be positive")
    ratio = float(in_size) / float(out_size)
    target = torch.arange(out_size, device=device, dtype=torch.float32)
    source_center = (target + 0.5) * ratio - 0.5
    source = torch.arange(in_size, device=device, dtype=torch.float32)
    distance = source[None, :] - source_center[:, None]
    # Widening the window for decimation gives the standard antialiased
    # Lanczos filter; the row normalization matches image-resize boundaries.
    scaled = distance / ratio if ratio > 1.0 else distance
    window = torch.sinc(scaled) * torch.sinc(scaled / float(radius))
    window = torch.where(scaled.abs() < float(radius), window,
                        torch.zeros_like(window))
    return (window / window.sum(dim=1, keepdim=True).clamp_min(1e-12)).to(dtype)


def _lanczos_downsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    if int(scale) < 1:
        raise ValueError("scale must be >= 1")
    if scale == 1:
        return x
    h, w = x.shape[-2:]
    if h % scale or w % scale:
        raise ValueError("Lanczos downsampling requires divisible dimensions")
    with torch.cuda.amp.autocast(enabled=False):
        x32 = x.float()
        wy = _lanczos_weights(h, h // scale, x.device, x32.dtype)
        wx = _lanczos_weights(w, w // scale, x.device, x32.dtype)
        y = torch.einsum("oi,bcij->bcoj", wy, x32)
        y = torch.einsum("xj,bcoj->bcox", wx, y)
    return y.to(dtype=x.dtype)


def _lanczos_adjoint(y: torch.Tensor, scale: int,
                     out_hw: tuple[int, int]) -> torch.Tensor:
    """Exact transpose of ``_lanczos_downsample`` for HQS data consistency."""
    h, w = out_hw
    if int(scale) < 1 or h % scale or w % scale:
        raise ValueError("invalid Lanczos adjoint dimensions")
    if y.shape[-2:] != (h // scale, w // scale):
        raise ValueError("Lanczos adjoint input size does not match out_hw")
    with torch.cuda.amp.autocast(enabled=False):
        y32 = y.float()
        wy = _lanczos_weights(h, h // scale, y.device, y32.dtype)
        wx = _lanczos_weights(w, w // scale, y.device, y32.dtype)
        x = torch.einsum("oi,bcoj->bcij", wy, y32)
        x = torch.einsum("xj,bcix->bcij", wx, x)
    return x.to(dtype=y.dtype)


class HoloForwardDet(nn.Module):
    """Differentiable deterministic part of the holographic degradation D."""

    def __init__(self, scale=4, cutoff=1.0, aa_sigma=0.5):
        super().__init__()
        self.scale = int(scale)
        self.cutoff = float(cutoff)
        self.aa_sigma = float(aa_sigma)

    def _mask(self, h, w, device, dtype):
        fy = torch.fft.fftfreq(h, device=device, dtype=torch.float32)[:, None]
        fx = torch.fft.fftfreq(w, device=device, dtype=torch.float32)[None, :]
        # The generator uses radius = sqrt((2fy)^2 + (2fx)^2).
        return ((2.0 * fy).square() + (2.0 * fx).square()).sqrt() <= self.cutoff

    def _transfer(self, h, w, device):
        fy = torch.fft.fftfreq(h, device=device, dtype=torch.float32)[:, None]
        fx = torch.fft.fftfreq(w, device=device, dtype=torch.float32)[None, :]
        aperture = self._mask(h, w, device, torch.float32)
        gaussian = torch.exp(-2.0 * math.pi ** 2 * self.aa_sigma ** 2 *
                             (fy.square() + fx.square()))
        return aperture * gaussian

    def _lowpass_and_antialias(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            f = torch.fft.fft2(x.float(), norm="ortho")
            f = f * self._transfer(x.shape[-2], x.shape[-1], x.device)
            x = torch.fft.ifft2(f, norm="ortho").real
        return x.to(dtype=x.dtype)

    def forward(self, x):
        if x.shape[-2] % self.scale or x.shape[-1] % self.scale:
            raise ValueError("HoloForwardDet requires dimensions divisible by scale")
        x = self._lowpass_and_antialias(x)
        return _lanczos_downsample(x, self.scale)

    def adjoint(self, y):
        x = _lanczos_adjoint(
            y, self.scale,
            (y.shape[-2] * self.scale, y.shape[-1] * self.scale))
        with torch.cuda.amp.autocast(enabled=False):
            f = torch.fft.fft2(x.float(), norm="ortho")
            f = f * self._transfer(x.shape[-2], x.shape[-1], x.device)
            x = torch.fft.ifft2(f, norm="ortho").real
        return x.to(dtype=y.dtype)


class HoloUnroll(nn.Module):
    """HQS-style physical data-consistency stages (P3)."""

    def __init__(self, channels=3, K=1, scale=4, learnable_step=True):
        super().__init__()
        self.K = int(K)
        self.D = HoloForwardDet(scale=scale)
        eta = torch.full((self.K,), 0.5)
        if learnable_step:
            self.eta = nn.Parameter(eta)
        else:
            self.register_buffer("eta", eta)
        self.prox = nn.ModuleList([
            HoloBlock(channels, 1, 2.0, False, "WithBias", scale=scale,
                      use_restore_branch=True, use_extrap_branch=True,
                      use_dirconv=True, use_multiscale=True,
                      use_nonlinear_v=True)
            for _ in range(self.K)
        ])

    def forward(self, x0, lr):
        x = x0
        for eta, prox in zip(self.eta, self.prox):
            residual = self.D.forward(x) - lr
            x = x - eta * self.D.adjoint(residual)
            x = prox(x)
        return x


class SRUpsample(nn.Sequential):
    """Pixel-shuffle reconstruction head for scale 2^n super-resolution."""

    def __init__(self, scale: int, channels: int):
        layers = []
        remaining = int(scale)
        if remaining < 1 or (remaining & (remaining - 1)):
            raise ValueError("HoloRestormer currently supports power-of-two scales")
        while remaining > 1:
            layers.extend([
                nn.Conv2d(channels, channels * 4, 3, 1, 1, bias=False),
                nn.PixelShuffle(2),
            ])
            remaining //= 2
        super().__init__(*layers)


class BandGatedSkip(nn.Module):
    """Suppress encoder skip channels dominated by out-of-band energy."""

    def __init__(self, dim, scale=4):
        super().__init__()
        self.split = HoloBandSplit(scale)
        hidden = max(int(dim) // 4, 8)
        self.proj = nn.Sequential(
            nn.Linear(int(dim) * 3, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, int(dim), bias=False), nn.Sigmoid())

    def forward(self, skip, dec):
        energy = self.split.band_energy(dec)
        b, c, _ = energy.shape
        weight = self.proj(energy.reshape(b, c * 3)).view(b, c, 1, 1)
        return skip * weight


class HoloRestormerV3(nn.Module):
    """Restormer-derived digital holographic super-resolution network."""

    def __init__(self, inp_channels=3, out_channels=3, scale=4, dim=48,
                 num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
                 heads=(1, 2, 4, 8), ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type="WithBias", use_homomorphic=True,
                 use_restore_branch=True, use_extrap_branch=True,
                 use_dirconv=True, use_multiscale=True, use_nonlinear_v=True,
                 use_skip_gate=True, use_unroll=False, unroll_K=1,
                 use_checkpoint=False, use_staged_head=True, **kwargs):
        super().__init__()
        del kwargs
        self.scale = int(scale)
        self.inp_channels = int(inp_channels)
        self.patch_embed = HoloStem(inp_channels, dim, use_homomorphic)

        block_args = dict(scale=self.scale, use_restore_branch=use_restore_branch,
                          use_extrap_branch=use_extrap_branch,
                          use_dirconv=use_dirconv, use_multiscale=use_multiscale,
                          use_nonlinear_v=use_nonlinear_v,
                          use_checkpoint=use_checkpoint)
        make = lambda channels, level: HoloBlock(
            dim=channels, num_heads=heads[level],
            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
            LayerNorm_type=LayerNorm_type, **block_args)
        self.encoder_level1 = nn.Sequential(*[
            make(dim, 0) for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            make(int(dim * 2), 1) for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim * 2))
        self.encoder_level3 = nn.Sequential(*[
            make(int(dim * 4), 2) for _ in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim * 4))
        self.latent = nn.Sequential(*[
            make(int(dim * 8), 3) for _ in range(num_blocks[3])])
        self.up4_3 = Upsample(int(dim * 8))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 8), int(dim * 4), 1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            make(int(dim * 4), 2) for _ in range(num_blocks[2])])
        self.up3_2 = Upsample(int(dim * 4))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 4), int(dim * 2), 1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            make(int(dim * 2), 1) for _ in range(num_blocks[1])])
        self.up2_1 = Upsample(int(dim * 2))
        self.decoder_level1 = nn.Sequential(*[
            make(int(dim * 2), 0) for _ in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[
            make(int(dim * 2), 0) for _ in range(num_refinement_blocks)])

        self.use_skip_gate = bool(use_skip_gate)
        self.skip_gate3 = BandGatedSkip(int(dim * 4), self.scale) if self.use_skip_gate else None
        self.skip_gate2 = BandGatedSkip(int(dim * 2), self.scale) if self.use_skip_gate else None
        self.skip_gate1 = BandGatedSkip(dim, self.scale) if self.use_skip_gate else None
        self.use_staged_head = bool(use_staged_head)
        if self.use_staged_head:
            # 两级 3 通道像素重排（v7 基线原样）
            self.output = nn.Conv2d(int(dim * 2), out_channels, 3, 1, 1, bias=bias)
            self.reconstruction = SRUpsample(self.scale, out_channels) \
                if self.scale > 1 else None
        else:
            # 官方 Restormer SR 头：一次出 out_channels*scale^2 通道，再 pixel_shuffle
            # （v6 那版 Restormer 就是这么做的：output = Conv2d(96, 48) + pixel_shuffle(4)）
            self.output = nn.Conv2d(int(dim * 2),
                                    out_channels * self.scale ** 2,
                                    3, 1, 1, bias=bias)
            self.reconstruction = (nn.PixelShuffle(self.scale)
                                   if self.scale > 1 else None)
        self.use_unroll = bool(use_unroll)
        self.unroll = HoloUnroll(out_channels, int(unroll_K), self.scale) \
            if self.use_unroll else None

    def forward(self, inp_img):
        if inp_img.shape[1] != self.inp_channels:
            raise ValueError(f"Expected {self.inp_channels} input channels, got {inp_img.shape[1]}")
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        out_enc_level2 = self.encoder_level2(self.down1_2(out_enc_level1))
        out_enc_level3 = self.encoder_level3(self.down2_3(out_enc_level2))
        latent = self.latent(self.down3_4(out_enc_level3))

        dec3 = self.up4_3(latent)
        skip3 = self.skip_gate3(out_enc_level3, dec3) if self.skip_gate3 else out_enc_level3
        dec3 = self.decoder_level3(self.reduce_chan_level3(torch.cat([dec3, skip3], 1)))
        dec2 = self.up3_2(dec3)
        skip2 = self.skip_gate2(out_enc_level2, dec2) if self.skip_gate2 else out_enc_level2
        dec2 = self.decoder_level2(self.reduce_chan_level2(torch.cat([dec2, skip2], 1)))
        dec1 = self.up2_1(dec2)
        skip1 = self.skip_gate1(out_enc_level1, dec1) if self.skip_gate1 else out_enc_level1
        dec1 = self.decoder_level1(torch.cat([dec1, skip1], 1))
        dec1 = self.refinement(dec1)
        residual = self.output(dec1)
        if self.reconstruction is not None:
            residual = self.reconstruction(residual)
            base = F.interpolate(inp_img, scale_factor=self.scale, mode="bicubic",
                                 align_corners=False)
            out = base + residual
        else:
            out = residual + inp_img
        if self.unroll is not None:
            base_hr = F.interpolate(inp_img, scale_factor=self.scale, mode="bicubic",
                                    align_corners=False)
            out = self.unroll(out if out.shape[-2:] == base_hr.shape[-2:] else base_hr,
                              inp_img)
        return out


class HoloRestormer(HoloRestormerV3):
    """Frozen-v7 baseline: Restormer-shaped x4 SR without v3 redesign flags.

    ``HoloRestormerV3`` remains available as an explicitly named research
    variant, but the v7 baseline config uses this class and must not silently
    enable homomorphic, band, directional, skip-gate, or unroll modules.
    """

    def __init__(self, **kwargs):
        baseline_defaults = {
            "use_homomorphic": False,
            "use_restore_branch": False,
            "use_extrap_branch": False,
            "use_dirconv": False,
            "use_multiscale": False,
            "use_nonlinear_v": False,
            "use_skip_gate": False,
            "use_unroll": False,
        }
        for key, value in baseline_defaults.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)



__all__ = [
    "HoloBandSplit", "HoloForwardDet", "HoloRestormer", "HoloRestormerV3",
    "HoloStem", "HoloUnroll", "spectral_mask",
]
