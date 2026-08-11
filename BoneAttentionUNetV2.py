from __future__ import annotations

import math
import weakref
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvINLeaky(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConvINLeaky(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvINLeaky(in_channels, out_channels),
            ConvINLeaky(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AdaptivePriorGuidanceModule(nn.Module):
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

        self.texture_k_raw = nn.Parameter(torch.log(torch.tensor(float(texture_k_init))))
        self.texture_mu_sigma = nn.Parameter(torch.tensor(float(texture_mu_init)))
        self.prior_fusion = nn.Sequential(
            nn.Conv3d(2, 2, kernel_size=1, bias=True),
            nn.BatchNorm3d(2),
            nn.Sigmoid(),
        )

    def set_ct_normalization_statistics(self, mean: float, std: float) -> None:
        if std <= 0:
            raise ValueError("CT standard deviation must be positive")
        self.ct_mean.fill_(float(mean))
        self.ct_std.fill_(float(std))

    def _recover_hu(self, x: torch.Tensor, raw_hu: Optional[torch.Tensor]) -> torch.Tensor:
        if raw_hu is not None:
            if raw_hu.ndim != 5 or raw_hu.shape[0] != x.shape[0]:
                raise ValueError("raw_hu must have shape [B, C, D, H, W] and match the batch size")
            return raw_hu[:, :1].float()
        return x[:, :1].float() * self.ct_std + self.ct_mean

    @staticmethod
    def _local_variance(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        padding = kernel_size // 2
        mean = F.avg_pool3d(x, kernel_size=kernel_size, stride=1, padding=padding)
        mean_sq = F.avg_pool3d(x.square(), kernel_size=kernel_size, stride=1, padding=padding)
        return (mean_sq - mean.square()).clamp_min(0.0)

    def forward(
        self,
        x_nnunet: torch.Tensor,
        raw_hu: Optional[torch.Tensor] = None,
        u_high: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        hu = self._recover_hu(x_nnunet, raw_hu)
        i_norm = torch.clamp(hu, float(self.hu_min), float(self.hu_max))
        i_norm = (i_norm - self.hu_min) / (self.hu_max - self.hu_min + 1e-6)

        batch_size = i_norm.shape[0]
        if u_high is None:
            q95 = torch.quantile(i_norm.reshape(batch_size, -1), self.upper_quantile, dim=1)
            u_high = q95.view(batch_size, 1, 1, 1, 1)
        else:
            u_high = torch.as_tensor(u_high, dtype=i_norm.dtype, device=i_norm.device)
            if u_high.ndim == 0:
                u_high = u_high.repeat(batch_size)
            if u_high.numel() != batch_size:
                raise ValueError("u_high must contain one value per batch element")
            u_high = u_high.reshape(batch_size, 1, 1, 1, 1)

        u_high = torch.minimum(u_high, self.u_high_cap)
        u_high = torch.maximum(u_high, self.u_low + 1e-4)

        p_bone = ((i_norm - self.u_low) / (u_high - self.u_low + 1e-6)).clamp(0.0, 1.0)
        if self.smooth_bone_prior:
            p_bone = F.avg_pool3d(p_bone, kernel_size=5, stride=1, padding=2)

        sigma2_local = self._local_variance(i_norm, kernel_size=3)
        k = F.softplus(self.texture_k_raw) + 1e-6
        p_texture = torch.sigmoid(k * (sigma2_local - self.texture_mu_sigma))

        priors = torch.cat([p_bone, p_texture], dim=1)
        priors = priors * self.prior_fusion(priors)

        return priors, {
            "i_norm": i_norm,
            "p_bone": p_bone,
            "p_texture": p_texture,
            "u_high": u_high,
            "texture_k": k.detach(),
            "texture_mu_sigma": self.texture_mu_sigma.detach(),
        }


class PriorGuidedFeatureModulator(nn.Module):
    def __init__(self, channels: int, prior_channels: int = 2, reduction: int = 4, token_grid: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        attn_dim = max(8, min(64, channels // 4))
        self.channels = int(channels)
        self.token_grid = int(token_grid)
        self.attn_dim = int(attn_dim)

        self.prior_proj = nn.Sequential(
            nn.Conv3d(prior_channels, channels, kernel_size=1, bias=True),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
        )
        self.gate_fc = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.q_proj = nn.Conv3d(channels, attn_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv3d(channels, attn_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.attn_out = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.fusion_fc = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def _pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        d, h, w = x.shape[2:]
        size = (min(self.token_grid, d), min(self.token_grid, h), min(self.token_grid, w))
        return F.adaptive_avg_pool3d(x, size)

    def forward(self, x: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        priors = F.interpolate(priors, size=x.shape[2:], mode="trilinear", align_corners=False)
        p_feat = self.prior_proj(priors)

        x_gap = F.adaptive_avg_pool3d(x, 1).flatten(1)
        p_gap = F.adaptive_avg_pool3d(p_feat, 1).flatten(1)
        channel_gate = self.gate_fc(torch.cat([x_gap, p_gap], dim=1))
        channel_gate = channel_gate.view(x.shape[0], self.channels, 1, 1, 1)
        p_feat = p_feat * channel_gate

        q = self._pool_tokens(self.q_proj(x)).flatten(2).transpose(1, 2)
        k = self._pool_tokens(self.k_proj(p_feat)).flatten(2)
        v_map = self._pool_tokens(self.v_proj(p_feat))
        v = v_map.flatten(2).transpose(1, 2)

        attn = torch.softmax(torch.matmul(q, k) / math.sqrt(float(self.attn_dim)), dim=-1)
        guided_tokens = torch.matmul(attn, v).transpose(1, 2)
        guided_low = guided_tokens.reshape(v_map.shape[0], self.channels, *v_map.shape[2:])
        guided_full = F.interpolate(guided_low, size=x.shape[2:], mode="trilinear", align_corners=False)
        guided = x + self.attn_out(guided_full) + p_feat

        guided_gap = F.adaptive_avg_pool3d(guided, 1).flatten(1)
        fusion_gate = self.fusion_fc(torch.cat([x_gap, guided_gap], dim=1))
        fusion_gate = fusion_gate.view(x.shape[0], self.channels, 1, 1, 1)
        return x * (1.0 - fusion_gate) + guided * fusion_gate


class VMA(nn.Module):
    def __init__(self, channels: int, factor: int = 32):
        super().__init__()
        groups = min(int(factor), int(channels))
        while channels % groups != 0:
            groups -= 1
        self.groups = groups
        channels_per_group = channels // groups

        self.softmax = nn.Softmax(dim=-1)
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.group_norm = nn.GroupNorm(channels_per_group, channels_per_group)
        self.cross_conv = nn.Conv3d(channels_per_group, channels_per_group, kernel_size=1, bias=True)
        self.local_conv = nn.Conv3d(channels_per_group, channels_per_group, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        g = self.groups
        cpg = c // g
        x_group = x.reshape(b * g, cpg, d, h, w)

        x_d = self.pool_d(x_group)
        x_h = self.pool_h(x_group).permute(0, 1, 3, 2, 4)
        x_w = self.pool_w(x_group).permute(0, 1, 4, 3, 2)
        cross = self.cross_conv(torch.cat([x_d, x_h, x_w], dim=2))
        x_d, x_h, x_w = torch.split(cross, [d, h, w], dim=2)

        f1 = self.group_norm(
            x_group
            * torch.sigmoid(x_d)
            * torch.sigmoid(x_h.permute(0, 1, 3, 2, 4))
            * torch.sigmoid(x_w.permute(0, 1, 4, 3, 2))
        )
        f2 = self.local_conv(x_group)

        w1 = self.softmax(self.global_pool(f1).reshape(b * g, 1, cpg))
        w2 = self.softmax(self.global_pool(f2).reshape(b * g, 1, cpg))
        raw_weight = torch.matmul(w1, f2.reshape(b * g, cpg, -1))
        raw_weight = raw_weight + torch.matmul(w2, f1.reshape(b * g, cpg, -1))
        w_final = torch.sigmoid(raw_weight).reshape(b * g, 1, d, h, w)
        spatial_weight = torch.sigmoid(w_final)
        return (x_group * spatial_weight).reshape(b, c, d, h, w)


class ConvMLP3D(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalAgg(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, kernel_size=9, padding=4, groups=dim, bias=True)
        self.norm1 = nn.InstanceNorm3d(dim, affine=True)
        self.pointwise = nn.Conv3d(dim, dim, kernel_size=1, bias=True)
        self.local_attn = nn.Conv3d(dim, dim, kernel_size=9, padding=4, groups=dim, bias=True)
        self.norm2 = nn.InstanceNorm3d(dim, affine=True)
        self.mlp = ConvMLP3D(dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pos = x + x * (torch.sigmoid(self.pos_embed(x)) - 0.5)
        x_attn = x_pos + x_pos * (
            torch.sigmoid(self.local_attn(self.pointwise(self.norm1(x_pos)))) - 0.5
        )
        return x_attn + x_attn * (torch.sigmoid(self.mlp(self.norm2(x_attn))) - 0.5)


class GlobalSparseAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = int(sr_ratio)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.up = (
            nn.ConvTranspose3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim, bias=True)
            if sr_ratio > 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        pooled = (
            F.avg_pool3d(x, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            if self.sr_ratio > 1
            else x
        )
        pd, ph, pw = pooled.shape[2:]

        tokens = pooled.flatten(2).transpose(1, 2)
        qkv = self.qkv(tokens).reshape(b, -1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, -1, c)
        out = self.proj(out).transpose(1, 2).reshape(b, c, pd, ph, pw)

        if self.sr_ratio > 1:
            out = self.up(out)
            if out.shape[2:] != (d, h, w):
                out = F.interpolate(out, size=(d, h, w), mode="trilinear", align_corners=False)
        return out


class SelfAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, sr_ratio: int = 1):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = GlobalSparseAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    @staticmethod
    def _apply_layer_norm(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        b, c, d, h, w = x.shape
        tokens = norm(x.flatten(2).transpose(1, 2))
        return tokens.transpose(1, 2).reshape(b, c, d, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        x = x + self.attn(self._apply_layer_norm(x, self.norm1))
        b, c, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, d, h, w)


class LKLGLBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, sr_ratio: int = 1):
        super().__init__()
        self.local = LocalAgg(dim, mlp_ratio=mlp_ratio) if sr_ratio > 1 else nn.Identity()
        self.self_attn = SelfAttn(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, sr_ratio=sr_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn(self.local(x))


class MixedScaleFeatureEnhancer(nn.Module):
    def __init__(self, channels: int = 320, expanded_channels: int = 384, depth: int = 3):
        super().__init__()
        heads_sparse = max(1, channels // 64)
        heads_full = max(1, expanded_channels // 64)
        self.sparse_blocks = nn.ModuleList(
            [LKLGLBlock(channels, heads_sparse, mlp_ratio=4.0, sr_ratio=2) for _ in range(depth)]
        )
        self.expand = DoubleConvINLeaky(channels, expanded_channels)
        self.full_blocks = nn.ModuleList(
            [LKLGLBlock(expanded_channels, heads_full, mlp_ratio=4.0, sr_ratio=1) for _ in range(depth)]
        )
        self.reduce = DoubleConvINLeaky(expanded_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.sparse_blocks:
            x = block(x)
        x = self.expand(x)
        for block in self.full_blocks:
            x = block(x)
        return self.reduce(x)


class PGMStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConvINLeaky(in_channels, out_channels)
        self.vma = VMA(out_channels, factor=32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vma(self.conv(x))


class _DecoderControl:
    def __init__(self, owner: "BoneAttentionUNetForNNUNet"):
        self._owner = weakref.proxy(owner)

    @property
    def deep_supervision(self) -> bool:
        return bool(self._owner.deep_supervision)

    @deep_supervision.setter
    def deep_supervision(self, enabled: bool) -> None:
        self._owner.deep_supervision = bool(enabled)


class BoneAttentionUNetForNNUNet(nn.Module):
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
        ct_mean: float = 0.0,
        ct_std: float = 1.0,
    ):
        super().__init__()
        expected_features = (32, 64, 128, 256, 320, 320)
        if tuple(feature_map_sizes) != expected_features:
            raise ValueError(f"PGMNet requires feature widths {expected_features}")
        if len(strides) != len(feature_map_sizes):
            raise ValueError("strides must contain one entry for each network stage")
        if tuple(strides[0]) != (1, 1, 1):
            raise ValueError("the first-stage stride must be (1, 1, 1)")

        self.deep_supervision = bool(deep_supervision)
        self.feature_map_sizes = tuple(int(v) for v in feature_map_sizes)
        self.strides = tuple(tuple(int(v) for v in s) for s in strides)
        self.apgm = AdaptivePriorGuidanceModule(ct_mean=ct_mean, ct_std=ct_std)

        self.encoders = nn.ModuleList()
        self.encoder_pgfms = nn.ModuleList()
        self.down_ops = nn.ModuleList()
        previous_channels = int(in_channels)
        for i, channels in enumerate(self.feature_map_sizes):
            self.encoders.append(PGMStage(previous_channels, channels))
            self.encoder_pgfms.append(PriorGuidedFeatureModulator(channels, prior_channels=2))
            previous_channels = channels
            if i < len(self.feature_map_sizes) - 1:
                stride = self.strides[i + 1]
                if stride == (2, 2, 2):
                    self.down_ops.append(nn.MaxPool3d(kernel_size=2, stride=2))
                else:
                    self.down_ops.append(nn.Conv3d(channels, channels, kernel_size=stride, stride=stride, bias=False))

        self.msfe = MixedScaleFeatureEnhancer(channels=320, expanded_channels=384, depth=3)

        self.up_ops = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.decoder_pgfms = nn.ModuleList()
        decoder_channels: List[int] = []
        for i in range(len(self.feature_map_sizes) - 1, 0, -1):
            in_ch = self.feature_map_sizes[i]
            out_ch = self.feature_map_sizes[i - 1]
            stride = self.strides[i]
            if stride == (2, 2, 2):
                self.up_ops.append(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2))
            else:
                self.up_ops.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=False),
                        nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=True),
                    )
                )
            self.decoders.append(PGMStage(out_ch * 2, out_ch))
            self.decoder_pgfms.append(PriorGuidedFeatureModulator(out_ch, prior_channels=2))
            decoder_channels.append(out_ch)

        self.final_conv = nn.Conv3d(self.feature_map_sizes[0], num_classes, kernel_size=1)
        self.deep_supervision_heads = nn.ModuleList(
            [nn.Conv3d(ch, num_classes, kernel_size=1) for ch in decoder_channels[:-1]]
        )
        self.decoder = _DecoderControl(self)

    def forward(
        self,
        x: torch.Tensor,
        raw_hu: Optional[torch.Tensor] = None,
        u_high: Optional[torch.Tensor] = None,
    ):
        priors, _ = self.apgm(x, raw_hu=raw_hu, u_high=u_high)

        skips: List[torch.Tensor] = []
        z = x
        for i, encoder in enumerate(self.encoders):
            z = self.encoder_pgfms[i](encoder(z), priors)
            skips.append(z)
            if i < len(self.down_ops):
                z = self.down_ops[i](z)

        z = self.msfe(z)

        decoder_features: List[torch.Tensor] = []
        for j, (up, decoder, pgfm) in enumerate(zip(self.up_ops, self.decoders, self.decoder_pgfms)):
            z = up(z)
            skip = skips[-2 - j]
            if z.shape[2:] != skip.shape[2:]:
                z = F.interpolate(z, size=skip.shape[2:], mode="trilinear", align_corners=False)
            z = pgfm(decoder(torch.cat([z, skip], dim=1)), priors)
            decoder_features.append(z)

        full_resolution_logits = self.final_conv(z)
        if not self.deep_supervision:
            return full_resolution_logits

        coarse_logits = [
            head(feature)
            for head, feature in zip(self.deep_supervision_heads, decoder_features[:-1])
        ]
        return [full_resolution_logits] + list(reversed(coarse_logits))


PGMNet = BoneAttentionUNetForNNUNet
