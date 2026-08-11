"""
PGMNet network implementation aligned to the final Supplementary Methods.

Manuscript-defined components
-----------------------------
1. APGM: Adaptive Prior Guidance Module
   - separate HU-domain normalization: clamp raw/reconstructed HU to [-200, 1800] and scale to [0, 1]
   - patient-specific bone-density prior:
       u_low = 0.18
       u_high = min(0.90, 95th percentile of I_norm)
       P_bone = clamp((I_norm-u_low)/(u_high-u_low), 0, 1)
   - local texture-complexity prior from 3x3x3 local variance:
       P_texture = sigmoid(k * (local_variance - mu_sigma))
     with learnable k and mu_sigma
   - optional 5x5x5 average smoothing of P_bone
   - concatenated priors are adaptively gated before PGFM
2. PGFM: Prior-Guided Feature Modulator
   - fully connected channel gate from target feature + prior summaries
   - prior-guided attention: Q from target features; K,V from prior features
   - gated fusion with the original feature map
3. VMA: Voxel-level Multi-dimensional Attention
   - grouped feature processing
   - D/H/W directional adaptive pooling and cross-dimensional interaction
   - GroupNorm gated branch + 3x3x3 local branch
   - bidirectional cross-attention weighting
4. MSFE: Mixed-Scale Feature Enhancer
   - LocalAgg with depthwise 9x9x9 positional/local attention, InstanceNorm and conv-MLP
   - sparse SelfAttn using AvgPool3d, scaled dot-product attention and depthwise ConvTranspose3d upsampling

Backbone
--------
Six 3D stages with channels (32, 64, 128, 256, 320, 320), 3x3x3 convolutions,
InstanceNorm3d, LeakyReLU and no dropout. The default strides yield spatial sizes
96 -> 48 -> 24 -> 12 -> 6 -> 3 for a 96^3 patch.

Important reproducibility note
------------------------------
This file is a manuscript-aligned reference implementation. It must not be described
as the exact historical training code unless the reported checkpoints/results were
actually produced with this implementation (or are shown to be state-dict/behavior
compatible). A 5-stage checkpoint is not expected to load strictly into this 6-stage model.
"""

from __future__ import annotations

import math
from functools import partial
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Backbone primitives: 3x3x3 Conv + InstanceNorm3d + LeakyReLU, no dropout
# -----------------------------------------------------------------------------
class ConvINLeaky(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: Tuple[int, int, int] = (1, 1, 1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConvINLeaky(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvINLeaky(in_ch, out_ch),
            ConvINLeaky(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# -----------------------------------------------------------------------------
# APGM: manuscript equations (1)-(4)
# -----------------------------------------------------------------------------
class AdaptivePriorGuidanceModule(nn.Module):
    """Adaptive Prior Guidance Module (APGM).

    The network input produced by nnU-Net is normally CT-normalized. To construct the
    manuscript-specified HU-domain priors without replacing standard nnU-Net input
    preprocessing, APGM reconstructs the clipped HU scale from the nnU-Net normalized
    first channel using the CT normalization mean/std supplied by the trainer.

    If the caller supplies raw_hu explicitly, raw_hu is used directly.
    """

    def __init__(
        self,
        ct_mean: float = 0.0,
        ct_std: float = 1.0,
        hu_min: float = -200.0,
        hu_max: float = 1800.0,
        u_low: float = 0.18,
        u_high_cap: float = 0.90,
        upper_quantile: float = 0.95,
        smooth_bone_prior: bool = True,
        texture_k_init: float = 10.0,
        texture_mu_init: float = 0.02,
    ):
        super().__init__()
        self.register_buffer("ct_mean", torch.tensor(float(ct_mean), dtype=torch.float32))
        self.register_buffer("ct_std", torch.tensor(max(float(ct_std), 1e-6), dtype=torch.float32))
        self.register_buffer("hu_min", torch.tensor(float(hu_min), dtype=torch.float32))
        self.register_buffer("hu_max", torch.tensor(float(hu_max), dtype=torch.float32))
        self.register_buffer("u_low", torch.tensor(float(u_low), dtype=torch.float32))
        self.register_buffer("u_high_cap", torch.tensor(float(u_high_cap), dtype=torch.float32))
        self.upper_quantile = float(upper_quantile)
        self.smooth_bone_prior = bool(smooth_bone_prior)

        # Manuscript: k and mu_sigma are learnable parameters.
        # Positive k is enforced with softplus; mu_sigma is unconstrained and learned.
        self.texture_k_raw = nn.Parameter(torch.tensor(float(texture_k_init)).log().clamp_min(-10.0))
        self.texture_mu_sigma = nn.Parameter(torch.tensor(float(texture_mu_init)))

        # Adaptive prior fusion gate; BatchNorm is restricted to this prior-fusion submodule,
        # while the segmentation backbone itself uses InstanceNorm3d as reported.
        self.prior_fusion = nn.Sequential(
            nn.Conv3d(2, 2, kernel_size=1, bias=True),
            nn.BatchNorm3d(2),
            nn.Sigmoid(),
        )

    def _recover_hu(self, x: torch.Tensor, raw_hu: Optional[torch.Tensor] = None) -> torch.Tensor:
        if raw_hu is not None:
            if raw_hu.ndim != 5:
                raise ValueError(f"raw_hu must be [B,C,D,H,W], got {tuple(raw_hu.shape)}")
            return raw_hu[:, :1].float()
        # Invert standard CT z-normalization for the first channel.
        return x[:, :1].float() * self.ct_std + self.ct_mean

    @staticmethod
    def _local_variance(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        pad = kernel_size // 2
        mean = F.avg_pool3d(x, kernel_size=kernel_size, stride=1, padding=pad)
        mean_sq = F.avg_pool3d(x * x, kernel_size=kernel_size, stride=1, padding=pad)
        return (mean_sq - mean * mean).clamp_min(0.0)

    def forward(
        self,
        x_nnunet: torch.Tensor,
        raw_hu: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        hu = self._recover_hu(x_nnunet, raw_hu=raw_hu)

        # Equation (1): independent APGM HU normalization.
        i_norm = torch.clamp(hu, float(self.hu_min), float(self.hu_max))
        i_norm = (i_norm - self.hu_min) / (self.hu_max - self.hu_min + 1e-6)

        # Patient-specific thresholds in normalized space.
        b = i_norm.shape[0]
        q95 = torch.quantile(i_norm.reshape(b, -1), self.upper_quantile, dim=1)
        q95 = q95.view(b, 1, 1, 1, 1)
        u_high = torch.minimum(q95, self.u_high_cap)
        u_high = torch.maximum(u_high, self.u_low + 1e-4)

        # Equation (2): bone-density prior.
        p_bone = ((i_norm - self.u_low) / (u_high - self.u_low + 1e-6)).clamp(0.0, 1.0)
        if self.smooth_bone_prior:
            p_bone = F.avg_pool3d(p_bone, kernel_size=5, stride=1, padding=2)

        # Equation (3): local texture-complexity prior, local variance in 3x3x3 neighborhood.
        sigma2_local = self._local_variance(i_norm, kernel_size=3)
        k = F.softplus(self.texture_k_raw) + 1e-6
        p_texture = torch.sigmoid(k * (sigma2_local - self.texture_mu_sigma))

        # Equation (4): concatenate patient-specific priors, then adaptively gate them.
        p = torch.cat([p_bone, p_texture], dim=1)
        p = p * self.prior_fusion(p)

        aux = {
            "i_norm": i_norm,
            "p_bone": p_bone,
            "p_texture": p_texture,
            "u_high": u_high,
            "texture_k": k.detach(),
            "texture_mu_sigma": self.texture_mu_sigma.detach(),
        }
        return p, aux


# -----------------------------------------------------------------------------
# PGFM: gated modulation + prior-guided Q/K/V attention + gated fusion
# -----------------------------------------------------------------------------
class PriorGuidedFeatureModulator(nn.Module):
    def __init__(self, channels: int, prior_channels: int = 2, reduction: int = 4, token_grid: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        attn_dim = max(8, min(64, channels // 4))
        self.channels = channels
        self.token_grid = int(token_grid)
        self.attn_dim = attn_dim

        self.prior_proj = nn.Sequential(
            nn.Conv3d(prior_channels, channels, 1, bias=True),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
        )

        # Fully connected channel gate from pooled target + prior features.
        self.gate_fc = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

        # Q from target features; K,V from prior-guided features.
        self.q_proj = nn.Conv3d(channels, attn_dim, 1, bias=False)
        self.k_proj = nn.Conv3d(channels, attn_dim, 1, bias=False)
        self.v_proj = nn.Conv3d(channels, channels, 1, bias=False)
        self.attn_out = nn.Conv3d(channels, channels, 1, bias=True)

        # Final gated fusion with the original target feature.
        self.fusion_fc = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def _pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        d, h, w = x.shape[2:]
        td, th, tw = min(self.token_grid, d), min(self.token_grid, h), min(self.token_grid, w)
        return F.adaptive_avg_pool3d(x, (td, th, tw))

    def forward(self, x: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        priors_r = F.interpolate(priors, size=x.shape[2:], mode="trilinear", align_corners=False)
        p_feat = self.prior_proj(priors_r)

        x_gap = F.adaptive_avg_pool3d(x, 1).flatten(1)
        p_gap = F.adaptive_avg_pool3d(p_feat, 1).flatten(1)
        channel_gate = self.gate_fc(torch.cat([x_gap, p_gap], dim=1)).view(x.shape[0], self.channels, 1, 1, 1)
        gated_prior = p_feat * channel_gate

        q = self._pool_tokens(self.q_proj(x)).flatten(2).transpose(1, 2)          # [B,N,Ca]
        k = self._pool_tokens(self.k_proj(gated_prior)).flatten(2)               # [B,Ca,N]
        v_map = self._pool_tokens(self.v_proj(gated_prior))
        v = v_map.flatten(2).transpose(1, 2)                                     # [B,N,C]

        attn = torch.softmax(torch.matmul(q, k) / math.sqrt(float(self.attn_dim)), dim=-1)
        attn_tokens = torch.matmul(attn, v).transpose(1, 2)
        attn_low = attn_tokens.reshape(v_map.shape[0], self.channels, *v_map.shape[2:])
        attn_full = F.interpolate(attn_low, size=x.shape[2:], mode="trilinear", align_corners=False)
        guided = x + self.attn_out(attn_full) + gated_prior

        guided_gap = F.adaptive_avg_pool3d(guided, 1).flatten(1)
        fusion_gate = self.fusion_fc(torch.cat([x_gap, guided_gap], dim=1)).view(x.shape[0], self.channels, 1, 1, 1)
        return x * (1.0 - fusion_gate) + guided * fusion_gate


# -----------------------------------------------------------------------------
# VMA: equations (6)-(12)
# -----------------------------------------------------------------------------
class VMA(nn.Module):
    def __init__(self, channels: int, factor: int = 32):
        super().__init__()
        # Use the largest valid group count <= factor so every reported channel width works.
        groups = min(int(factor), channels)
        while channels % groups != 0:
            groups -= 1
        self.groups = groups
        cpg = channels // groups

        self.softmax = nn.Softmax(dim=-1)
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.gn = nn.GroupNorm(cpg, cpg)
        self.conv1x1 = nn.Conv3d(cpg, cpg, 1, bias=True)
        self.conv3x3 = nn.Conv3d(cpg, cpg, 3, padding=1, bias=True)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        g = self.groups
        cpg = c // g

        # Eq. (6)
        xg = x.reshape(b * g, cpg, d, h, w)

        # Eq. (7): directional pooling.
        xd = self.pool_d(xg)                                     # [BG,Cg,D,1,1]
        xh = self.pool_h(xg).permute(0, 1, 3, 2, 4)              # [BG,Cg,H,1,1]
        xw = self.pool_w(xg).permute(0, 1, 4, 3, 2)              # [BG,Cg,W,1,1]

        # Eq. (8): cross-dimensional interaction and redistribution.
        fcross = self.conv1x1(torch.cat([xd, xh, xw], dim=2))
        xd_new, xh_new, xw_new = torch.split(fcross, [d, h, w], dim=2)

        # Eq. (9): multidimensional gated enhancement + GroupNorm.
        f1 = self.gn(
            xg
            * torch.sigmoid(xd_new)
            * torch.sigmoid(xh_new.permute(0, 1, 3, 2, 4))
            * torch.sigmoid(xw_new.permute(0, 1, 4, 3, 2))
        )

        # Eq. (11) local contextual branch; Eq. (10)/(11) channel weights.
        f2 = self.conv3x3(xg)
        w1 = self.softmax(self.global_pool(f1).reshape(b * g, 1, cpg))
        w2 = self.softmax(self.global_pool(f2).reshape(b * g, 1, cpg))
        f2_flat = f2.reshape(b * g, cpg, -1)
        f1_flat = f1.reshape(b * g, cpg, -1)

        # Eq. (12): bidirectional cross-attention and voxel weighting.
        wfinal = torch.matmul(w1, f2_flat) + torch.matmul(w2, f1_flat)
        wfinal = wfinal.reshape(b * g, 1, d, h, w)
        spatial_weight = torch.sigmoid(wfinal)
        self.last_attn_weights = spatial_weight.detach()
        return (xg * spatial_weight).reshape(b, c, d, h, w)


# -----------------------------------------------------------------------------
# MSFE: LocalAgg + sparse SelfAttn, equations (13)-(16)
# -----------------------------------------------------------------------------
class ConvMLP3D(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalAgg(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 9, padding=4, groups=dim, bias=True)
        self.norm1 = nn.InstanceNorm3d(dim, affine=True)
        self.conv1 = nn.Conv3d(dim, dim, 1, bias=True)
        self.conv2 = nn.Conv3d(dim, dim, 1, bias=True)
        self.local_attn = nn.Conv3d(dim, dim, 9, padding=4, groups=dim, bias=True)
        self.norm2 = nn.InstanceNorm3d(dim, affine=True)
        self.mlp = ConvMLP3D(dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Supplement Eq. (13) and immediately surrounding LocalAgg equations.
        xpos = x + x * (torch.sigmoid(self.pos_embed(x)) - 0.5)
        xattn = xpos + xpos * (
            torch.sigmoid(self.conv2(self.local_attn(self.conv1(self.norm1(xpos))))) - 0.5
        )
        xout = xattn + xattn * (torch.sigmoid(self.mlp(self.norm2(xattn))) - 0.5)
        return xout


class GlobalSparseAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = int(sr_ratio)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        if self.sr_ratio > 1:
            self.up = nn.ConvTranspose3d(
                dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio, groups=dim, bias=True
            )
        else:
            self.up = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        if self.sr_ratio > 1:
            # Eq. (14): hierarchical average-pooling before Q/K/V.
            pooled = F.avg_pool3d(x, kernel_size=self.sr_ratio, stride=self.sr_ratio)
        else:
            pooled = x
        pd, ph, pw = pooled.shape[2:]

        tokens = pooled.flatten(2).transpose(1, 2)                  # [B,N,C]
        qkv = self.qkv(tokens).reshape(b, -1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)                                      # [B,H,N,Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Eq. (15): scaled dot-product attention.
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, -1, c)
        out = self.proj(out)
        out = out.transpose(1, 2).reshape(b, c, pd, ph, pw)

        # Eq. (16): depthwise transposed-convolution upsampling to original resolution.
        if self.sr_ratio > 1:
            out = self.up(out)
            if out.shape[2:] != (d, h, w):
                out = F.interpolate(out, size=(d, h, w), mode="trilinear", align_corners=False)
        return out


class SelfAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, sr_ratio: int = 1):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim, bias=True)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = GlobalSparseAttn(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    @staticmethod
    def _apply_ln(x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
        b, c, d, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = ln(t)
        return t.transpose(1, 2).reshape(b, c, d, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        x = x + self.attn(self._apply_ln(x, self.norm1))

        b, c, d, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = t + self.mlp(self.norm2(t))
        return t.transpose(1, 2).reshape(b, c, d, h, w)


class LKLGLBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, sr_ratio: int = 1):
        super().__init__()
        self.local = LocalAgg(dim, mlp_ratio=mlp_ratio) if sr_ratio > 1 else nn.Identity()
        self.self_attn = SelfAttn(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, sr_ratio=sr_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn(self.local(x))


class MixedScaleFeatureEnhancer(nn.Module):
    """MSFE bottleneck consistent with the Supplementary Note.

    The first stack uses local aggregation + sparse global attention (sr_ratio=2).
    Features are then expanded to 384 channels and refined at full resolution
    (sr_ratio=1), before being reduced back to 320 channels.
    """

    def __init__(self, channels: int = 320, expanded_channels: int = 384, depth: int = 3):
        super().__init__()
        h1 = max(1, channels // 64)
        h2 = max(1, expanded_channels // 64)
        self.sparse_blocks = nn.ModuleList(
            [LKLGLBlock(channels, num_heads=h1, mlp_ratio=4.0, sr_ratio=2) for _ in range(depth)]
        )
        self.expand = DoubleConvINLeaky(channels, expanded_channels)
        self.full_blocks = nn.ModuleList(
            [LKLGLBlock(expanded_channels, num_heads=h2, mlp_ratio=4.0, sr_ratio=1) for _ in range(depth)]
        )
        self.reduce = DoubleConvINLeaky(expanded_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.sparse_blocks:
            x = blk(x)
        x = self.expand(x)
        for blk in self.full_blocks:
            x = blk(x)
        return self.reduce(x)


# -----------------------------------------------------------------------------
# Six-stage PGMNet encoder-decoder
# -----------------------------------------------------------------------------
class PGMStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_vma: bool = True):
        super().__init__()
        self.conv = DoubleConvINLeaky(in_ch, out_ch)
        self.vma = VMA(out_ch, factor=32) if use_vma else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vma(self.conv(x))


class BoneAttentionUNetForNNUNet(nn.Module):
    """PGMNet manuscript-aligned network used by nnUNetTrainerBoneAttention."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        feature_map_sizes: Tuple[int, ...] = (32, 64, 128, 256, 320, 320),
        strides: Tuple[Tuple[int, int, int], ...] = (
            (1, 1, 1),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
        ),
        deep_supervision: bool = True,
        max_ds_outputs: int = 4,
        ct_mean: float = 0.0,
        ct_std: float = 1.0,
    ):
        super().__init__()
        expected = (32, 64, 128, 256, 320, 320)
        if tuple(feature_map_sizes) != expected:
            raise ValueError(
                f"Manuscript-aligned PGMNet requires six stages {expected}, got {tuple(feature_map_sizes)}"
            )
        if len(strides) == len(feature_map_sizes) - 1:
            strides = ((1, 1, 1),) + tuple(strides)
        if len(strides) != len(feature_map_sizes):
            raise ValueError("strides must contain one entry per stage, including stage-0 stride (1,1,1)")

        self.deep_supervision = bool(deep_supervision)
        self.max_ds_outputs = int(max_ds_outputs)
        self.feature_map_sizes = tuple(feature_map_sizes)
        self.strides = tuple(tuple(int(v) for v in s) for s in strides)

        # APGM always enabled in the final architecture.
        self.apgm = AdaptivePriorGuidanceModule(ct_mean=ct_mean, ct_std=ct_std)

        # Encoder: PGMNet stages + PGFM.
        self.encoders = nn.ModuleList()
        self.pgfms_enc = nn.ModuleList()
        self.down_ops = nn.ModuleList()
        prev = in_channels
        for i, ch in enumerate(self.feature_map_sizes):
            self.encoders.append(PGMStage(prev, ch, use_vma=True))
            self.pgfms_enc.append(PriorGuidedFeatureModulator(ch, prior_channels=2))
            prev = ch
            if i < len(self.feature_map_sizes) - 1:
                s = self.strides[i + 1]
                self.down_ops.append(
                    nn.MaxPool3d(kernel_size=s, stride=s)
                    if s == (2, 2, 2)
                    else nn.Conv3d(ch, ch, kernel_size=s, stride=s, bias=False)
                )

        # MSFE at the deepest 320-channel stage.
        self.msfe = MixedScaleFeatureEnhancer(channels=320, expanded_channels=384, depth=3)

        # Decoder + PGFM.
        self.up_ops = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pgfms_dec = nn.ModuleList()
        decoder_channels: List[int] = []
        for i in range(len(self.feature_map_sizes) - 1, 0, -1):
            in_ch = self.feature_map_sizes[i]
            out_ch = self.feature_map_sizes[i - 1]
            s = self.strides[i]
            if s == (2, 2, 2):
                self.up_ops.append(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2))
            else:
                self.up_ops.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=s, mode="trilinear", align_corners=False),
                        nn.Conv3d(in_ch, out_ch, 1, bias=True),
                    )
                )
            self.decoders.append(PGMStage(out_ch * 2, out_ch, use_vma=True))
            self.pgfms_dec.append(PriorGuidedFeatureModulator(out_ch, prior_channels=2))
            decoder_channels.append(out_ch)

        self.final_conv = nn.Conv3d(self.feature_map_sizes[0], num_classes, 1)

        # Full-resolution output + up to 3 auxiliary outputs = 4 deep-supervision outputs.
        if self.deep_supervision:
            aux_channels = decoder_channels[: max(0, self.max_ds_outputs - 1)]
            self.ds_convs = nn.ModuleList([nn.Conv3d(ch, num_classes, 1) for ch in aux_channels])
        else:
            self.ds_convs = None

        # nnU-Net compatibility proxy.
        class _DecoderProxy:
            pass
        self.decoder = _DecoderProxy()
        self.decoder.deep_supervision = self.deep_supervision

    def forward(self, x: torch.Tensor, raw_hu: Optional[torch.Tensor] = None):
        priors, _ = self.apgm(x, raw_hu=raw_hu)

        skips: List[torch.Tensor] = []
        z = x
        for i, enc in enumerate(self.encoders):
            z = enc(z)
            z = self.pgfms_enc[i](z, priors)
            skips.append(z)
            if i < len(self.down_ops):
                z = self.down_ops[i](z)

        z = self.msfe(z)

        decoder_feats: List[torch.Tensor] = []
        # skip the deepest feature because z already represents it
        for j, (up, dec, pgfm) in enumerate(zip(self.up_ops, self.decoders, self.pgfms_dec)):
            z = up(z)
            skip = skips[-2 - j]
            if z.shape[2:] != skip.shape[2:]:
                z = F.interpolate(z, size=skip.shape[2:], mode="trilinear", align_corners=False)
            z = dec(torch.cat([z, skip], dim=1))
            z = pgfm(z, priors)
            decoder_feats.append(z)

        full = self.final_conv(z)
        if not self.deep_supervision or self.ds_convs is None:
            return full

        outputs = [full]
        # decoder_feats are deepest-to-shallowest. Generate auxiliary outputs from the
        # first (coarser) decoder stages. Dynamic loss resizes targets to each output.
        for feat, head in zip(decoder_feats, self.ds_convs):
            outputs.append(head(feat))
        return outputs


# Alias used by external code / manuscript naming.
PGMNet = BoneAttentionUNetForNNUNet


if __name__ == "__main__":
    # Lightweight architectural smoke test independent of nnU-Net.
    torch.manual_seed(0)
    net = BoneAttentionUNetForNNUNet(
        in_channels=1,
        num_classes=2,
        feature_map_sizes=(32, 64, 128, 256, 320, 320),
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        deep_supervision=True,
        ct_mean=300.0,
        ct_std=500.0,
    )
    # 32^3 is sufficient for code-path testing while keeping memory modest.
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        y = net(x)
    print([tuple(t.shape) for t in y] if isinstance(y, list) else tuple(y.shape))
