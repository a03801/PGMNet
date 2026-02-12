from typing import List, Tuple, Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.layers import DropPath
# ---------------------------
# Basic blocks
# ---------------------------

# class SEBlock(nn.Module):
#     def __init__(self, channels: int, reduction: int = 16):
#         super().__init__()
#         hidden = max(4, channels // reduction)
#         self.pool = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, channels),
#             nn.Sigmoid()
#         )
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         b, c, _, _, _ = x.shape
#         y = self.pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1, 1)
#         return x * y

# class SpatialAttention3D(nn.Module):
#     def __init__(self, kernel_size: int = 7):
#         super().__init__()
#         padding = (kernel_size - 1) // 2
#         self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
#         self.bn = nn.BatchNorm3d(1)
#         self.act = nn.Sigmoid()
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         attn = self.conv(torch.cat([avg_out, max_out], dim=1))
#         attn = self.bn(attn)
#         return self.act(attn)

# class CBAM3D(nn.Module):
#     def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
#         super().__init__()
#         self.channel_att = SEBlock(channels, reduction)
#         self.spatial_att = SpatialAttention3D(kernel_size)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.channel_att(x)
#         s = self.spatial_att(x)
#         return x * s
# Voxel-level Multi-dimensional Attention
class VMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(VMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool3d((1, 1, 1))   # 全局平均池化
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        # GroupNorm 注意：num_groups = channels // groups，num_channels = channels // groups
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv3d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv3d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, d, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, d, h, w)  # [b*g, c//g, d, h, w]

        # 沿各个维度的 pool
        x_d = self.pool_d(group_x)    # [b*g, c//g, d,1,1]  #(64,2,16,1,1)
        x_h = self.pool_h(group_x).permute(0,1,3,2,4)   # [b*g, c//g,1,h,1]   #(64,2,1,64,1)
        x_w = self.pool_w(group_x).permute(0,1,4,3,2)   # [b*g, c//g,1,1,w]   #(64,2,1,1,64)
        # 拼接特征 (扩展到相同维度)
        # hw = self.conv1x1(torch.cat([
        #     x_d.expand(-1, -1, d, 1, 1),
        #     x_h.expand(-1, -1, h, 1, 1),
        #     x_w.expand(-1, -1, w, 1, 1)
        # ], dim=2))   # 在 depth 维拼接
        hw = self.conv1x1(torch.cat([
            x_d,
            x_h,
            x_w
        ], dim=2))   # 在 depth 维拼接

        # 拆分回 d,h,w 三部分
        x_d_new, x_h_new, x_w_new = torch.split(hw, [d, h, w], dim=2)

        # 通道内增强
        x1 = self.gn(group_x * x_d_new.sigmoid() * 
                                x_h_new.permute(0,1,3,2,4) .sigmoid() * 
                                x_w_new.permute(0,1,4,3,2) .sigmoid() )
        x2 = self.conv3x3(group_x)

        # 注意力权重计算
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # [b*g,1,c//g]
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # [b*g,c//g,dhw]

        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, d, h, w)
        
        self.last_attn_weights = weights.sigmoid().detach()

        return (group_x * weights.sigmoid()).reshape(b, c, d, h, w)
    

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvUtr(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, kernel=3):
        super(ConvUtr, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv3d(ch_in, ch_in, kernel_size=(kernel, kernel, kernel), groups=ch_in, padding=(kernel // 2, kernel // 2, kernel // 2)),
                    nn.GELU(),
                    nn.InstanceNorm3d(ch_in)
                )),
                Residual(nn.Sequential(
                    nn.Conv3d(ch_in, ch_in * 4, kernel_size=(1, 1, 1)),
                    nn.GELU(),
                    nn.InstanceNorm3d(ch_in * 4),
                    nn.Conv3d(ch_in * 4, ch_in, kernel_size=(1, 1, 1)),
                    nn.GELU(),
                    nn.InstanceNorm3d(ch_in)
                )),
            ) for i in range(depth)]
        )
        self.up = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x
    
class enbedding(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_cbam: bool = False):
        super().__init__()
        # self.stem = nn.Sequential(
        #     nn.Conv3d(1, dims[0], kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.InstanceNorm3d(dims[0]),
        #     nn.ReLU(inplace=True)
        # )
        self.layer = ConvUtr(in_ch, out_ch, depth=1, kernel=3)
        # self.layer2 = ConvUtr(in_ch, out_ch, depth=1, kernel=3)
        # self.layer3 = ConvUtr(in_ch, out_ch, depth=3, kernel=7)
        # self.convblock = 
        self.VMA = VMA(out_ch) if use_cbam else None
        # self.down = nn.MaxPool3d(kernel_size=2, stride=2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        if self.VMA:
            x = self.VMA(x)

        return x
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 3, padding=1, groups=in_features)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 3, padding=1, groups=in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalSparseAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = sr_ratio
        if self.sr > 1:
            self.sampler = nn.AvgPool3d(1, sr_ratio)
            kernel_size = sr_ratio
            self.LocalProp = nn.ConvTranspose3d(dim, dim, kernel_size, stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.upsample = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x, H: int, W: int, D: int):
        B, N, C = x.shape
        if self.sr > 1.:
            x = x.transpose(1, 2).reshape(B, C, H, W, D)
            # x = x.transpose(1, 2).reshape(B, C, H, W, D)
            x = self.sampler(x)
            x = x.flatten(2).transpose(1, 2)

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.sr > 1:
            x = x.permute(0, 2, 1).reshape(B, C, int(H / self.sr), int(W / self.sr), int(D / self.sr))
            x = self.LocalProp(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocalAgg(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 9, padding=4, groups=dim)
        self.norm1 = nn.InstanceNorm3d(dim)
        self.conv1 = nn.Conv3d(dim, dim, 1)
        self.conv2 = nn.Conv3d(dim, dim, 1)
        self.attn = nn.Conv3d(dim, dim, 9, padding=4, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.InstanceNorm3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.sg = nn.Sigmoid()

    def forward(self, x):
        x = x + x * (self.sg(self.pos_embed(x)) - 0.5)
        x = x + x * (self.sg(self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))) - 0.5)
        x = x + x * (self.sg(self.drop_path(self.mlp(self.norm2(x)))) - 0.5)
        return x


class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1.):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalSparseAttn(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W, D = x.shape
        # B, N, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, D))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W, D)
        # x = x.transpose(1, 2).reshape(B, N, D, H, W)
        return x


class LKLGLBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1.):
        super().__init__()

        if sr_ratio > 1:
            self.LocalAgg = LocalAgg(dim, mlp_ratio, drop, drop_path, act_layer)
        else:
            self.LocalAgg = nn.Identity()

        self.SelfAttn = SelfAttn(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer,
                                 norm_layer, sr_ratio)

    def forward(self, x):
        x = self.LocalAgg(x)  #(1,64,16,12,12)
        x = self.SelfAttn(x)
        return x

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
# class ASPP3D(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  dilations: Sequence[int] = (1, 6, 12, 18),
#                  use_bn: bool = True,
#                  use_gap: bool = True):
#         super().__init__()
#         branches = nn.ModuleList()
#         for d in dilations:
#             branches.append(nn.Sequential(
#                 nn.Conv3d(in_channels, out_channels, 3, padding=d, dilation=d, bias=not use_bn),
#                 nn.BatchNorm3d(out_channels) if use_bn else nn.Identity(),
#                 nn.ReLU(inplace=True)
#             ))
#         self.branches = branches
#         self.use_gap = use_gap
#         if use_gap:
#             self.gap = nn.Sequential(
#                 nn.AdaptiveAvgPool3d(1),
#                 nn.Conv3d(in_channels, out_channels, 1, bias=False),
#                 nn.BatchNorm3d(out_channels) if use_bn else nn.Identity(),
#                 nn.ReLU(inplace=True)
#             )
#         total = (len(dilations) + (1 if use_gap else 0)) * out_channels
#         self.project = nn.Sequential(
#             nn.Conv3d(total, out_channels, 1, bias=False),
#             nn.BatchNorm3d(out_channels) if use_bn else nn.Identity(),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         size = x.shape[2:]
#         feats = [b(x) for b in self.branches]
#         if self.use_gap:
#             gap = self.gap(x)
#             gap = F.interpolate(gap, size=size, mode='trilinear', align_corners=False)
#             feats.append(gap)
#         return self.project(torch.cat(feats, dim=1))

# class DoubleConv(nn.Module):
#     def __init__(self, in_ch: int, out_ch: int, use_cbam: bool = False):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#         self.VMA = VMA(out_ch) if use_cbam else None
#         # self.cbam = CBAM3D(out_ch) if use_cbam else None

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.block(x)
#         if self.VMA:
#             x = self.VMA(x)
#         # if self.cbam:
#         #     x = self.cbam(x)
#         return x

class FPN3D(nn.Module):
    def __init__(self, in_channels_list: List[int], out_ch: int):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv3d(c, out_ch, 1) for c in in_channels_list])
        self.smooth = nn.ModuleList([nn.Conv3d(out_ch, out_ch, 3, padding=1) for _ in in_channels_list])
    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:], mode='trilinear', align_corners=False)
            laterals[i] = laterals[i] + up
        return [s(l) for s, l in zip(self.smooth, laterals)]

# ---------------------------
# New: in-model CT normalization and prior-gated attention
# ---------------------------

class CTLinearNormalizer(nn.Module):
    """
    Clip HU to [hu_min, hu_max] then linearly scale to [0, 1].
    """
    def __init__(self, hu_min: float = -200.0, hu_max: float = 1800.0, apply_to_all_channels: bool = True):
        super().__init__()
        self.register_buffer("hu_min", torch.tensor(float(hu_min)))
        self.register_buffer("hu_max", torch.tensor(float(hu_max)))
        self.apply_to_all_channels = bool(apply_to_all_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_to_all_channels or x.shape[1] == 1:
            vmin, vmax = self.hu_min, self.hu_max
            x = torch.clamp(x, vmin.item(), vmax.item())
            x = (x - vmin) / (vmax - vmin + 1e-6)
            return x
        c0 = x[:, :1]
        vmin, vmax = self.hu_min, self.hu_max
        c0 = torch.clamp(c0, vmin.item(), vmax.item())
        c0 = (c0 - vmin) / (vmax - vmin + 1e-6)
        return torch.cat([c0, x[:, 1:]], dim=1)

class PriorFusionGate3D(nn.Module):
    """
    Fuse prior maps (e.g., bone, artifact) into a single spatial gate A in [0,1].
    gate = sigmoid(Conv1x1([priors])) -> [B,1,D,H,W]
    """
    def __init__(self, num_priors: int):
        super().__init__()
        assert num_priors >= 1
        self.fuse = nn.Sequential(
            nn.Conv3d(num_priors, 1, kernel_size=1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
    def forward(self, priors: torch.Tensor) -> torch.Tensor:
        return self.fuse(priors)

def _quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    b = x.shape[0]
    flat = x.reshape(b, -1)
    try:
        qv = torch.quantile(flat, q, dim=1, keepdim=True)
    except Exception:
        k = max(1, int(flat.shape[1] * (1 - q)))
        vals, _ = torch.topk(flat, k=k, dim=1, largest=True)
        qv = vals.min(dim=1, keepdim=True)[0]
    return qv.view(b, 1, 1, 1, 1)

# ---------------------------
# Main network with new options
# ---------------------------

class BoneAttentionUNetForNNUNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 feature_map_sizes: Tuple[int, ...] = (32, 64, 128, 256, 320),
                 strides: Tuple[Tuple[int,int,int], ...] = ((1,1,1),(2,2,2),(2,2,2),(2,2,2),(2,2,2)),
                 use_cbam: bool = True,
                 use_aspp: bool = False,
                 use_fpn: bool = False,
                 deep_supervision: bool = True,
                 max_ds_outputs: int = 4,
                 # New options:
                 use_inmodel_norm: bool = False,
                 norm_hu_min: float = -200.0,
                 norm_hu_max: float = 1800.0,
                 use_bone_prior: bool = False,
                 use_artifact_prior: bool = False,
                 prior_alpha: float = 0.5,
                 prior_apply_to: str = "all",           # "enc" | "dec" | "all"
                 prior_kernel_size: int = 3,
                 # Bone prior thresholds
                 bone_hu_low: float = 180.0,           # fixed low = 180 HU
                 bone_hu_high: float = 1500.0,         # used when dynamic_high disabled
                 # Dynamic high settings
                 use_dynamic_high: bool = True,
                 dynamic_high_p: float = 0.995,        # p99.5
                 dynamic_high_min_hu: float = 1200.0,  # guard rail
                 dynamic_high_max_hu: float = 1800.0):
        super().__init__()
        if len(strides) == len(feature_map_sizes) - 1:
            strides = ((1,1,1),) + strides
        assert len(strides) >= len(feature_map_sizes)
        self.deep_supervision = deep_supervision
        self.use_fpn = use_fpn
        self.use_aspp = use_aspp
        self.max_ds_outputs = max_ds_outputs

        # New flags/params
        self.use_inmodel_norm = bool(use_inmodel_norm)
        self.normalizer = CTLinearNormalizer(norm_hu_min, norm_hu_max, apply_to_all_channels=True) if self.use_inmodel_norm else None
        self.use_bone_prior = bool(use_bone_prior)
        self.use_artifact_prior = bool(use_artifact_prior)
        self.prior_alpha = float(prior_alpha)
        self.prior_apply_to = str(prior_apply_to).lower()
        self.prior_kernel_size = int(prior_kernel_size)

        # Store HU ranges as buffers
        self.register_buffer("hu_clip_min", torch.tensor(float(norm_hu_min)))
        self.register_buffer("hu_clip_max", torch.tensor(float(norm_hu_max)))
        self.register_buffer("bone_hu_low", torch.tensor(float(bone_hu_low)))
        self.register_buffer("bone_hu_high", torch.tensor(float(bone_hu_high)))

        # Dynamic high controls
        self.use_dynamic_high = bool(use_dynamic_high)
        self.register_buffer("dynamic_high_p", torch.tensor(float(dynamic_high_p)))
        self.register_buffer("dynamic_high_min_hu", torch.tensor(float(dynamic_high_min_hu)))
        self.register_buffer("dynamic_high_max_hu", torch.tensor(float(dynamic_high_max_hu)))

        num_priors = (1 if self.use_bone_prior else 0) + (1 if self.use_artifact_prior else 0)
        self.prior_gate: Optional[PriorFusionGate3D] = PriorFusionGate3D(num_priors) if num_priors >= 1 else None

        # Encoder
        self.encoders = nn.ModuleList()
        self.down_ops = nn.ModuleList()
        prev = in_channels
        for i, c in enumerate(feature_map_sizes):
            self.encoders.append(enbedding(prev, c, use_cbam))
            prev = c
            if i < len(feature_map_sizes) - 1:
                s = strides[i + 1]
                if s == (2,2,2):
                    self.down_ops.append(nn.MaxPool3d(2))
                else:
                    self.down_ops.append(nn.Conv3d(c, c, kernel_size=s, stride=s, padding=0, bias=False))

        # bottleneck_ch = feature_map_sizes[-1]
        # self.bottleneck = ASPP3D(bottleneck_ch, bottleneck_ch) if use_aspp else DoubleConv(bottleneck_ch, bottleneck_ch, use_cbam)
        self.lklgl_bottleneck = nn.ModuleList([
            LKLGLBlock(
                dim=320, num_heads=320 // 64, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), sr_ratio=2)
            for _ in range(3)])

        self.transformer_bottleneck = nn.ModuleList([
            LKLGLBlock(
                dim=384, num_heads=384 // 64, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), sr_ratio=1)
            for _ in range(3)])
        self.expend_dims = conv_block(ch_in=320, ch_out=384)
        self.reduce_dims = conv_block(ch_in=384, ch_out=320)
        # Decoder
        self.up_ops = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(feature_map_sizes) - 1, 0, -1):
            in_ch = feature_map_sizes[i]
            out_ch = feature_map_sizes[i - 1]
            s = strides[i]
            if s == (2,2,2):
                self.up_ops.append(nn.ConvTranspose3d(in_ch, out_ch, 2, 2))
            else:
                self.up_ops.append(nn.Sequential(
                    nn.Upsample(scale_factor=s, mode='trilinear', align_corners=False),
                    nn.Conv3d(in_ch, out_ch, 1)
                ))
            self.decoders.append(enbedding(out_ch * 2, out_ch, use_cbam))

        if use_fpn:
            self.fpn = FPN3D(list(feature_map_sizes), feature_map_sizes[0])
        self.final_conv = nn.Conv3d(feature_map_sizes[0], num_classes, 1)

        if deep_supervision:
            aux_channels_full = [feature_map_sizes[-2], feature_map_sizes[-3], feature_map_sizes[-4], feature_map_sizes[-5] if len(feature_map_sizes)>=5 else None]
            aux_channels_full = [c for c in aux_channels_full if c is not None]
            aux_channels = aux_channels_full[:-1]
            aux_channels = aux_channels[:max(0, max_ds_outputs - 1)]
            self.ds_convs = nn.ModuleList([nn.Conv3d(ch, num_classes, 1) for ch in aux_channels])
        else:
            self.ds_convs = None

    # ----- helpers -----
    def _map_hu_to_unit(self, hu_value: torch.Tensor) -> torch.Tensor:
        return (hu_value - self.hu_clip_min) / (self.hu_clip_max - self.hu_clip_min + 1e-6)

    def _compute_artifact_prior(self, x01: torch.Tensor) -> torch.Tensor:
        k = max(1, self.prior_kernel_size)
        mu = F.avg_pool3d(x01, kernel_size=k, stride=1, padding=k//2)
        mu2 = F.avg_pool3d(x01 * x01, kernel_size=k, stride=1, padding=k//2)
        var = (mu2 - mu * mu).clamp_min(0.0)
        # normalize by per-scan 90th percentile
        b = var.shape[0]
        flat = var.reshape(b, -1)
        try:
            q90 = torch.quantile(flat, 0.90, dim=1, keepdim=True).view(b,1,1,1,1)
        except Exception:
            ktop = max(1, int(flat.shape[1] * 0.10))
            vals, _ = torch.topk(flat, k=ktop, dim=1, largest=True)
            q90 = vals.min(dim=1, keepdim=True)[0].view(b,1,1,1,1)
        prior = (var / (var + q90 + 1e-6)).clamp(0.0, 1.0)
        return prior

    def _get_bone_low_high_u(self, x01: torch.Tensor):
        """
        Return (low_u, high_u) in unit space [0,1].
        low_u: fixed from 180 HU
        high_u: per-scan dynamic p99.5 with guard [1200,1800] HU if enabled; else fixed.
        Shapes: low_u and high_u -> [B,1,1,1,1]
        """
        b = x01.shape[0]
        low_u_scalar = self._map_hu_to_unit(self.bone_hu_low).clamp(0.0, 1.0)
        if self.use_dynamic_high:
            # p = 0.995 quantile in unit space
            flat = x01.reshape(b, -1)
            try:
                q = torch.quantile(flat, float(self.dynamic_high_p.item()), dim=1, keepdim=True).view(b,1,1,1,1)
            except Exception:
                k = max(1, int(flat.shape[1] * (1 - float(self.dynamic_high_p.item()))))
                vals, _ = torch.topk(flat, k=k, dim=1, largest=True)
                q = vals.min(dim=1, keepdim=True)[0].view(b,1,1,1,1)
            # clamp by HU guard rails, mapped to unit
            guard_min_u = self._map_hu_to_unit(self.dynamic_high_min_hu).clamp(0.0, 1.0)
            guard_max_u = self._map_hu_to_unit(self.dynamic_high_max_hu).clamp(0.0, 1.0)
            high_u = torch.clamp(q, min=guard_min_u.item(), max=guard_max_u.item())
            low_u = low_u_scalar.view(1,1,1,1,1).expand_as(high_u)
        else:
            high_u_scalar = self._map_hu_to_unit(self.bone_hu_high).clamp(0.0, 1.0)
            high_u = high_u_scalar.view(1,1,1,1,1).expand(b,1,1,1,1)
            low_u = low_u_scalar.view(1,1,1,1,1).expand_as(high_u)
        # ensure high > low
        high_u = torch.maximum(high_u, low_u + 1e-3)
        return low_u, high_u

    def _compute_bone_prior(self, x01: torch.Tensor) -> torch.Tensor:
        """
        x01: [B,1,D,H,W] in [0,1]
        """
        low_u, high_u = self._get_bone_low_high_u(x01)
        ramp = (x01 - low_u) / (high_u - low_u + 1e-6)
        p = ramp.clamp(0.0, 1.0)
        if self.prior_kernel_size and self.prior_kernel_size > 1:
            k = self.prior_kernel_size
            p = F.avg_pool3d(p, kernel_size=k, stride=1, padding=k//2)
        return p

    def _maybe_build_prior_stack(self, x_in01: torch.Tensor) -> Optional[torch.Tensor]:
        if self.prior_gate is None:
            return None
        x0 = x_in01[:, :1]
        priors = []
        if self.use_bone_prior:
            priors.append(self._compute_bone_prior(x0))
        if self.use_artifact_prior:
            priors.append(self._compute_artifact_prior(x0))
        if not priors:
            return None
        return torch.cat(priors, dim=1)

    def _apply_prior_gate(self, feat: torch.Tensor, prior_stack_fullres: torch.Tensor) -> torch.Tensor:
        gate = self.prior_gate(prior_stack_fullres)  # [B,1,D,H,W] at input res
        if gate.shape[2:] != feat.shape[2:]:
            gate = F.interpolate(gate, size=feat.shape[2:], mode='trilinear', align_corners=False)
        mod = self.prior_alpha * gate + (1.0 - self.prior_alpha)
        return feat * mod

    # ----- Forward -----
    def forward(self, x: torch.Tensor):
        # Optional HU->unit normalization inside the model
        x_in = self.normalizer(x) if self.normalizer is not None else x

        # Prior maps computed at input resolution from normalized channel 0
        prior_stack_fullres = None
        if self.prior_gate is not None:
            prior_stack_fullres = self._maybe_build_prior_stack(x_in)

        # Encoder
        enc_feats = []
        out = x_in
        for i, enc in enumerate(self.encoders):
            out = enc(out)
            if (prior_stack_fullres is not None) and (self.prior_apply_to in ("enc", "all")):
                out = self._apply_prior_gate(out, prior_stack_fullres)
            enc_feats.append(out)
            if i < len(self.down_ops):
                out = self.down_ops[i](out)

        # Bottleneck
        # bottleneck = self.bottleneck(enc_feats[-1])
        for blk in self.lklgl_bottleneck:
            bottleneck = blk(enc_feats[-1])
        bottleneck = self.expend_dims(bottleneck)

        for blk in self.transformer_bottleneck:
            bottleneck = blk(bottleneck)
        bottleneck = self.reduce_dims(bottleneck)
        # Decoder
        dec_feats = []
        dec = bottleneck
        for i in range(len(self.up_ops)):
            dec = self.up_ops[i](dec)
            skip = enc_feats[-(i + 2)]
            if skip.shape[2:] != dec.shape[2:]:
                dec = F.interpolate(dec, size=skip.shape[2:], mode='trilinear', align_corners=False)
            dec = torch.cat([skip, dec], dim=1)
            dec = self.decoders[i](dec)
            if (prior_stack_fullres is not None) and (self.prior_apply_to in ("dec", "all")):
                dec = self._apply_prior_gate(dec, prior_stack_fullres)
            dec_feats.append(dec)

        # FPN fusion (optional)
        full_res_feat = dec_feats[-1]
        if self.use_fpn:
            fpn_feats = self.fpn(enc_feats)
            high_res_fpn = fpn_feats[0]
            if high_res_fpn.shape[2:] != full_res_feat.shape[2:]:
                high_res_fpn = F.interpolate(high_res_fpn, size=full_res_feat.shape[2:], mode='trilinear', align_corners=False)
            full_res_feat = full_res_feat + high_res_fpn

        main_logits = self.final_conv(full_res_feat)

        if not self.training or not self.deep_supervision or self.ds_convs is None:
            return main_logits

        aux_feats = dec_feats[:-1]
        outputs = [main_logits]
        for head, f in zip(self.ds_convs, aux_feats):
            outputs.append(head(f))
        return outputs

if __name__ == "__main__":
    # Demo
    model = BoneAttentionUNetForNNUNet(
        in_channels=1,
        num_classes=2,
        feature_map_sizes=(32,64,128,256,320),
        strides=((1,1,1),(2,2,2),(2,2,2),(2,2,2),(2,2,2)),
        deep_supervision=True,
        use_inmodel_norm=True,          # HU->unit normalization [-200,1800]
        norm_hu_min=-200.0,
        norm_hu_max=1800.0,
        use_bone_prior=True,
        use_artifact_prior=True,
        prior_alpha=0.5,
        prior_apply_to="all",
        prior_kernel_size=3,
        bone_hu_low=180.0,              # fixed low
        use_dynamic_high=True,          # dynamic high with guard rails
        dynamic_high_p=0.995,
        dynamic_high_min_hu=1200.0,
        dynamic_high_max_hu=1800.0
    )
    model.train()
    x = torch.randn(2,1,128,128,128) * 500 + 200.0  # pseudo HU
    y = model(x)
    print([t.shape for t in y] if isinstance(y, list) else y.shape)
    

# /home/li/anaconda3/envs/nnunet/lib/python3.9/site-packages/dynamic_network_architectures/architectures/BoneAttentionUNetV2.py
