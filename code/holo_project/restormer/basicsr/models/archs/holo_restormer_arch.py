"""Restormer adapted for the fixed holographic LR x4 -> HR task.

The ``attention_mode`` switch is the only architectural difference between
the quick decision experiments:

* ``mdta``: the original Restormer channel-transposed attention;
* ``band``: radial Fourier-band attention with the same input/output feature
  width and the same SR head.

This file intentionally has no dataset or machine-specific paths.
"""

from __future__ import annotations

import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_3d(x):
    return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


def _to_4d(x, h, w):
    return x.reshape(x.shape[0], h, w, x.shape[-1]).permute(0, 3, 1, 2)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    def __init__(self, dim, mode="WithBias"):
        super().__init__()
        self.body = BiasFreeLayerNorm(dim) if mode == "BiasFree" else WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, expansion=2.66, bias=False):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1,
                                groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class MDTA(nn.Module):
    """The original Restormer multi-Dconv head transposed attention."""

    def __init__(self, dim, heads, bias=False):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1,
                                groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        ch = c // self.heads
        q = q.reshape(b, self.heads, ch, h * w)
        k = k.reshape(b, self.heads, ch, h * w)
        v = v.reshape(b, self.heads, ch, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = (attn.softmax(dim=-1) @ v).reshape(b, c, h, w)
        return self.project_out(out)


class BandAttention(nn.Module):
    """Restormer attention with the channel axis replaced by Fourier bands.

    This follows the design in ``全息超分-基线选型与改造设计.md``:

    1. transform the feature map to the complex rFFT domain;
    2. split the spectrum into ``n_bands`` radial complex sub-bands;
    3. pool each complex sub-band into one token, retaining real *and*
       imaginary parts (the phase is therefore not discarded);
    4. run the MDTA-style multi-head attention over the band-token axis;
    5. merge the complex band tokens back to the spectrum and apply iFFT.

    ``Conv2d(dim, 3 * dim, 1)`` is kept for q/k/v as in the document.  Since a
    complex token has two real components, a small real/imag encoder and
    decoder surround that projection; these do not change the number of band
    tokens or the attention axis.
    """

    def __init__(self, dim, heads, n_bands=6, bias=False):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        if int(n_bands) < 2:
            raise ValueError("n_bands must be at least 2")
        self.dim = dim
        self.heads = heads
        self.n_bands = int(n_bands)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.complex_encode = nn.Conv2d(2 * dim, dim, 1, bias=bias)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.complex_decode = nn.Conv2d(dim, 2 * dim, 1, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def split_radial(self, spectrum, h, w, n_bands=None):
        """Return masked complex radial sub-bands and their frequency index."""
        n_bands = self.n_bands if n_bands is None else int(n_bands)
        wf = spectrum.shape[-1]
        # Use physical FFT frequencies rather than image-grid coordinates.  In
        # particular, the rFFT x-axis contains only non-negative frequencies.
        fy = torch.fft.fftfreq(h, d=1.0, device=spectrum.device).view(h, 1)
        fx = torch.fft.rfftfreq(w, d=1.0, device=spectrum.device).view(1, wf)
        radius = torch.sqrt(fy.square() + fx.square())
        radius = radius / radius.amax().clamp_min(torch.finfo(radius.dtype).eps)
        idx = torch.floor(radius * n_bands).long().clamp_max(n_bands - 1)
        masks = F.one_hot(idx, num_classes=n_bands).permute(2, 0, 1)
        masks = masks.to(dtype=spectrum.real.dtype)
        bands = [spectrum * masks[i].unsqueeze(0).unsqueeze(0)
                 for i in range(n_bands)]
        return bands, idx

    def _band_tokens(self, bands, idx):
        """Pool each complex sub-band into B x (2C) x K x 1 tokens."""
        tokens = []
        for band_id, band in enumerate(bands):
            # Count the radial-bin locations, not non-zero signal values.  A
            # valid sub-band may legitimately contain exact spectral zeros.
            mask = idx == band_id
            count = mask.sum().clamp_min(1).to(dtype=band.real.dtype)
            real = band.real.sum(dim=(-2, -1)) / count
            imag = band.imag.sum(dim=(-2, -1)) / count
            tokens.append(torch.cat([real, imag], dim=1))
        return torch.stack(tokens, dim=2).unsqueeze(-1)

    def attention(self, qkv):
        """Apply multi-head attention over K=n_bands tokens."""
        b, _, k, _ = qkv.shape
        q, key, value = qkv.chunk(3, dim=1)
        ch = q.shape[1] // self.heads
        q = q.squeeze(-1).reshape(b, self.heads, ch, k).transpose(-1, -2)
        key = key.squeeze(-1).reshape(b, self.heads, ch, k).transpose(-1, -2)
        value = value.squeeze(-1).reshape(b, self.heads, ch, k).transpose(-1, -2)
        q = F.normalize(q, dim=-1)
        key = F.normalize(key, dim=-1)
        weights = (q @ key.transpose(-2, -1)) * self.temperature
        out = (weights.softmax(dim=-1) @ value).transpose(-1, -2)
        return out.reshape(b, self.dim, k, 1)

    def merge_bands(self, tokens, idx, h, w):
        """Broadcast complex band tokens back to their radial frequency bins."""
        b, _, k, _ = tokens.shape
        real, imag = self.complex_decode(tokens).squeeze(-1).chunk(2, dim=1)
        complex_tokens = torch.complex(real, imag)  # B,C,K
        masks = F.one_hot(idx.reshape(-1), num_classes=k)
        masks = masks.reshape(h, w, k).permute(2, 0, 1)
        masks = masks.to(device=tokens.device, dtype=complex_tokens.real.dtype)
        return (complex_tokens.unsqueeze(-1).unsqueeze(-1) *
                masks.unsqueeze(0).unsqueeze(0)).sum(dim=2)

    def forward(self, x):
        b, c, h, w = x.shape
        spectrum = torch.fft.rfft2(x, norm="ortho")
        bands, idx = self.split_radial(spectrum, h, w, self.n_bands)
        tokens = self._band_tokens(bands, idx)
        tokens = self.complex_encode(tokens)
        qkv = self.qkv(tokens)
        out = self.attention(qkv)
        out = self.merge_bands(out, idx, h, spectrum.shape[-1])
        out = torch.fft.irfft2(out, s=(h, w), norm="ortho")
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, expansion=2.66, bias=False,
                 layer_norm="WithBias", attention_mode="mdta", n_bands=6):
        super().__init__()
        self.norm1 = LayerNorm2d(dim, layer_norm)
        self.attn = MDTA(dim, heads, bias) if attention_mode == "mdta" else BandAttention(dim, heads, n_bands, bias)
        self.norm2 = LayerNorm2d(dim, layer_norm)
        self.ffn = FeedForward(dim, expansion, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class Downsample(nn.Module):
    def __init__(self, feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(feat, feat // 2, 3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(feat, feat * 2, 3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class HoloRestormer(nn.Module):
    """Restormer U-Net with a fixed x4 super-resolution reconstruction head."""

    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
                 heads=(1, 2, 4, 8), ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type="WithBias",
                 attention_mode="mdta", n_bands=6, scale=4):
        super().__init__()
        if scale != 4:
            raise ValueError("The fixed holographic experiment is LR x4 -> HR")
        self.scale = scale
        block = lambda feat, head, count: nn.Sequential(*[
            TransformerBlock(feat, head, ffn_expansion_factor, bias,
                             LayerNorm_type, attention_mode, n_bands)
            for _ in range(count)
        ])
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, padding=1, bias=bias)
        self.encoder_level1 = block(dim, heads[0], num_blocks[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = block(dim * 2, heads[1], num_blocks[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = block(dim * 4, heads[2], num_blocks[2])
        self.down3_4 = Downsample(dim * 4)
        self.latent = block(dim * 8, heads[3], num_blocks[3])
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = block(dim * 4, heads[2], num_blocks[2])
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = block(dim * 2, heads[1], num_blocks[1])
        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = block(dim * 2, heads[0], num_blocks[0])
        self.refinement = block(dim * 2, heads[0], num_refinement_blocks)
        self.output = nn.Conv2d(dim * 2, out_channels * scale * scale, 3,
                                padding=1, bias=bias)

    def forward(self, inp_img):
        e1 = self.encoder_level1(self.patch_embed(inp_img))
        e2 = self.encoder_level2(self.down1_2(e1))
        e3 = self.encoder_level3(self.down2_3(e2))
        latent = self.latent(self.down3_4(e3))
        d3 = self.reduce_chan_level3(torch.cat([self.up4_3(latent), e3], 1))
        d3 = self.decoder_level3(d3)
        d2 = self.reduce_chan_level2(torch.cat([self.up3_2(d3), e2], 1))
        d2 = self.decoder_level2(d2)
        d1 = self.decoder_level1(torch.cat([self.up2_1(d2), e1], 1))
        d1 = self.refinement(d1)
        residual = F.pixel_shuffle(self.output(d1), self.scale)
        base = F.interpolate(inp_img, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return residual + base


# Convenient names for generated configs and explicit baseline semantics.
RestormerSR = HoloRestormer

