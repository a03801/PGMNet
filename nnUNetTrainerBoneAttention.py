from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

try:
    from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
except Exception:  # keep import error informative at trainer construction time
    PolyLRScheduler = None

try:
    from dynamic_network_architectures.architectures.BoneAttentionUNetV2 import BoneAttentionUNetForNNUNet
except Exception:
    try:
        from BoneAttentionUNetV2 import BoneAttentionUNetForNNUNet
    except Exception:
        from BoneAttentionUNetV2_manuscript_aligned import BoneAttentionUNetForNNUNet


os.environ.setdefault("NNUNET_COMPILE", "0")
os.environ.setdefault("nnUNet_compile", "0")
os.environ.setdefault("MPLBACKEND", "Agg")


# -----------------------------------------------------------------------------
# Target and loss utilities
# -----------------------------------------------------------------------------
def to_class_map(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 4:
        return t.long()
    if t.ndim == 5:
        if t.shape[1] == 1:
            return t[:, 0].long()
        return torch.argmax(t, dim=1).long()
    raise ValueError(f"Unsupported target tensor shape {tuple(t.shape)}")


def select_full_res_target(target):
    if isinstance(target, torch.Tensor):
        return to_class_map(target), None, None
    if isinstance(target, (list, tuple)):
        sizes = [int(t.shape[-3] * t.shape[-2] * t.shape[-1]) for t in target]
        idx = int(torch.tensor(sizes).argmax().item())
        return to_class_map(target[idx]), target, idx
    raise TypeError(f"Unsupported target type {type(target)}")


def resize_class_map(class_map: torch.Tensor, size):
    if class_map.shape[1:] == tuple(size):
        return class_map
    return F.interpolate(class_map.unsqueeze(1).float(), size=size, mode="nearest")[:, 0].long()


def soft_dice_loss_from_class(
    logits: torch.Tensor,
    class_map: torch.Tensor,
    smooth: float = 1e-5,
    ignore_background: bool = True,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    num_classes = probs.shape[1]
    one_hot = F.one_hot(class_map.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = torch.sum(probs * one_hot, dim=dims)
    denom = torch.sum(probs + one_hot, dim=dims)
    dice_pc = (2.0 * inter + smooth) / (denom + smooth)
    if ignore_background and num_classes > 1:
        dice_pc = dice_pc[1:]
    return 1.0 - dice_pc.mean()


class CustomDynMultiScaleLoss(nn.Module):
    """Deep-supervision CE + foreground soft-Dice with 50-epoch weight annealing."""

    def __init__(
        self,
        trainer_ref,
        init_ce_w: float = 0.3,
        init_dc_w: float = 0.7,
        target_ce_w: float = 0.5,
        target_dc_w: float = 0.5,
        warmup_epochs: int = 50,
    ):
        super().__init__()
        self.trainer = trainer_ref
        self.init_ce_w = float(init_ce_w)
        self.init_dc_w = float(init_dc_w)
        self.target_ce_w = float(target_ce_w)
        self.target_dc_w = float(target_dc_w)
        self.warmup_epochs = int(warmup_epochs)

    def _weights(self) -> Tuple[float, float]:
        epoch = float(getattr(self.trainer, "current_epoch", 0))
        p = min(1.0, epoch / max(1.0, float(self.warmup_epochs)))
        ce = self.init_ce_w + (self.target_ce_w - self.init_ce_w) * p
        dc = self.init_dc_w + (self.target_dc_w - self.init_dc_w) * p
        s = ce + dc
        return ce / s, dc / s

    def forward(self, outputs, target):
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        full_res_map, _, _ = select_full_res_target(target)

        # Standard geometrically decreasing deep-supervision weights.
        raw = [0.5 ** i for i in range(len(outputs))]
        total_raw = sum(raw)
        ds_weights = [w / total_raw for w in raw]

        total_ce = outputs[0].new_tensor(0.0)
        total_dc = outputs[0].new_tensor(0.0)
        for logit, w in zip(outputs, ds_weights):
            scaled = resize_class_map(full_res_map, logit.shape[2:])
            total_ce = total_ce + w * F.cross_entropy(logit, scaled)
            total_dc = total_dc + w * soft_dice_loss_from_class(logit, scaled, ignore_background=True)

        ce_w, dc_w = self._weights()
        return ce_w * total_ce + dc_w * total_dc


# -----------------------------------------------------------------------------
# Helpers for exact six-stage plan and CT normalization statistics
# -----------------------------------------------------------------------------
EXPECTED_FEATURES = (32, 64, 128, 256, 320, 320)
EXPECTED_STRIDES = (
    (1, 1, 1),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 2),
)


def _num_classes_from_dataset_json(dataset_json: Dict[str, Any]) -> int:
    labels = dataset_json.get("labels", {}) if isinstance(dataset_json, dict) else {}
    if isinstance(labels, dict) and labels:
        try:
            return max(int(v) for v in labels.values()) + 1
        except Exception:
            pass
    return 2


def _search_mean_std(obj: Any) -> Optional[Tuple[float, float]]:
    """Recursively find CT foreground mean/std in a plans/fingerprint-like mapping."""
    if isinstance(obj, dict):
        # Typical nnU-Net fingerprint keys.
        if "mean" in obj and "std" in obj:
            try:
                mean, std = float(obj["mean"]), float(obj["std"])
                if std > 0:
                    return mean, std
            except Exception:
                pass
        # Prefer channel 0 and CT-related containers where present.
        preferred = [
            "foreground_intensity_properties_per_channel",
            "intensityproperties",
            "intensity_properties",
            "0",
            0,
        ]
        for key in preferred:
            if key in obj:
                found = _search_mean_std(obj[key])
                if found is not None:
                    return found
        for value in obj.values():
            found = _search_mean_std(value)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = _search_mean_std(value)
            if found is not None:
                return found
    return None


def _extract_ct_mean_std(*objects: Any) -> Tuple[float, float]:
    for obj in objects:
        if obj is None:
            continue
        # inspect common attributes without depending on a specific nnU-Net minor release
        candidates = [obj]
        for attr in ("plans", "dataset_json", "foreground_intensity_properties_per_channel"):
            if hasattr(obj, attr):
                candidates.append(getattr(obj, attr))
        for candidate in candidates:
            found = _search_mean_std(candidate)
            if found is not None:
                return found

    # Explicit fallback for installations where the static architecture API does not expose plans.
    # These values should be copied from the model's dataset_fingerprint/plans so APGM can invert
    # the standard nnU-Net CT z-normalization before applying its own [-200, 1800] HU mapping.
    mean = float(os.environ.get("PGMNET_CT_MEAN", "0.0"))
    std = float(os.environ.get("PGMNET_CT_STD", "1.0"))
    if std <= 0:
        raise ValueError("PGMNET_CT_STD must be > 0")
    return mean, std


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class nnUNetTrainerBoneAttention(nnUNetTrainer):
    """Final manuscript-aligned PGMNet trainer."""

    def __init__(self, plans, configuration, fold, dataset_json, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)

        # Explicit manuscript training configuration.
        self.initial_lr = 0.01
        self.weight_decay = 3e-5
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.oversample_foreground_percent = 0.33
        self.enable_deep_supervision = True

        self.disable_validation = False
        self.perform_validation_every_x_epochs = 1
        self.last_dice_per_class = None
        self.last_mean_fg = None

    @staticmethod
    def build_network_architecture(*args, **kwargs):
        """Build exactly the six-stage architecture reported in the Supplement.

        Supports both common nnU-Net v2 architecture-builder call signatures.
        """
        net_in_ch = None
        net_num_classes = None
        enable_ds = True
        ct_mean, ct_std = 0.0, 1.0

        # Newer static signature:
        # architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
        # num_input_channels, num_output_channels, enable_deep_supervision
        if len(args) >= 6 and isinstance(args[1], dict):
            _, arch_conf, _, net_in_ch, net_num_classes, enable_ds = args[:6]
            ct_mean, ct_std = _extract_ct_mean_std(arch_conf)

            # Fail visibly if the supplied plan contradicts the manuscript rather than
            # silently truncating six stages to five (the defect in the old trainer).
            fms = arch_conf.get("features_per_stage") or arch_conf.get("num_features_per_stage")
            if fms is not None and tuple(int(v) for v in fms) != EXPECTED_FEATURES:
                raise RuntimeError(
                    f"nnUNetPlans feature widths {tuple(fms)} do not match manuscript {EXPECTED_FEATURES}. "
                    "Regenerate/use the reported six-stage plan; do not truncate it."
                )

        # Older/alternative signature:
        # plans_manager, dataset_json, configuration_manager, num_input_channels, enable_ds
        elif len(args) >= 5:
            plans_manager, dataset_json, configuration_manager, net_in_ch, enable_ds = args[:5]
            net_num_classes = _num_classes_from_dataset_json(dataset_json)
            ct_mean, ct_std = _extract_ct_mean_std(plans_manager, configuration_manager, dataset_json)
        else:
            raise RuntimeError(
                "nnUNetTrainerBoneAttention.build_network_architecture received an unsupported nnU-Net call signature"
            )

        # APGM relies on reversing standard CT z-normalization. If static API did not expose
        # mean/std, allow exact model-specific values to be supplied explicitly.
        if "PGMNET_CT_MEAN" in os.environ:
            ct_mean = float(os.environ["PGMNET_CT_MEAN"])
        if "PGMNET_CT_STD" in os.environ:
            ct_std = float(os.environ["PGMNET_CT_STD"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = BoneAttentionUNetForNNUNet(
            in_channels=int(net_in_ch),
            num_classes=int(net_num_classes),
            feature_map_sizes=EXPECTED_FEATURES,
            strides=EXPECTED_STRIDES,
            deep_supervision=bool(enable_ds),
            max_ds_outputs=4,
            ct_mean=ct_mean,
            ct_std=ct_std,
        ).to(device)
        return net

    def configure_optimizers(self):
        # Manuscript: SGD, initial LR=0.01, Nesterov momentum=0.99, weight decay=3e-5,
        # polynomial learning-rate scheduler.
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        if PolyLRScheduler is None:
            raise ImportError(
                "Could not import nnunetv2.training.lr_scheduler.polylr.PolyLRScheduler. "
                "Install the nnU-Net v2 version used for the manuscript (2.5.2)."
            )
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, scheduler

    def _build_loss(self):
        # Build the reported dynamic multiscale CE + foreground soft-Dice loss from the start,
        # instead of replacing the loss only after training initialization.
        return CustomDynMultiScaleLoss(
            trainer_ref=self,
            init_ce_w=0.3,
            init_dc_w=0.7,
            target_ce_w=0.5,
            target_dc_w=0.5,
            warmup_epochs=50,
        )

    def on_train_start(self):
        super().on_train_start()
        self.set_deep_supervision_enabled(True)

    def should_validate_now(self):
        if self.disable_validation:
            return False
        # Supplement: validation every epoch.
        return True

    def validation_step(self, batch: dict):
        if self.disable_validation:
            return None
        self.network.eval()
        data = batch["data"].to(self.device, non_blocking=True, dtype=torch.float32)
        target = batch["target"]

        with torch.no_grad():
            out = self.network(data)
            if isinstance(out, (list, tuple)):
                out = out[0]
            pred = torch.argmax(torch.softmax(out, dim=1), dim=1, keepdim=True)

        full_map, _, _ = select_full_res_target(target)
        gt = full_map.unsqueeze(1).to(pred.device)
        num_classes = out.shape[1]
        pred_onehot = torch.zeros(
            (pred.shape[0], num_classes, *pred.shape[2:]), device=pred.device, dtype=torch.int
        )
        pred_onehot.scatter_(1, pred.long(), 1)
        axes = list(range(2, pred_onehot.ndim))
        tp, fp, fn, tn = get_tp_fp_fn_tn(pred_onehot, gt.long(), axes=axes, mask=None)
        return {
            "tp": tp.sum(0),
            "fp": fp.sum(0),
            "fn": fn.sum(0),
            "tn": tn.sum(0),
            "num_samples": pred.shape[0],
        }

    def on_validation_epoch_end(self, val_outputs):
        if self.disable_validation or not val_outputs:
            return
        val_outputs = [v for v in val_outputs if v is not None]
        if not val_outputs:
            return

        tp = torch.stack([v["tp"] for v in val_outputs], 0).sum(0)
        fp = torch.stack([v["fp"] for v in val_outputs], 0).sum(0)
        fn = torch.stack([v["fn"] for v in val_outputs], 0).sum(0)
        eps = 1e-8
        dice_per_class = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        mean_fg = dice_per_class[1:].mean().item() if dice_per_class.numel() > 1 else dice_per_class.mean().item()
        self.last_dice_per_class = dice_per_class.detach().cpu().numpy()
        self.last_mean_fg = mean_fg

    def set_deep_supervision_enabled(self, enabled: bool):
        self.enable_deep_supervision = bool(enabled)
        if hasattr(self, "network"):
            if hasattr(self.network, "deep_supervision"):
                self.network.deep_supervision = bool(enabled)
            if hasattr(self.network, "decoder") and hasattr(self.network.decoder, "deep_supervision"):
                self.network.decoder.deep_supervision = bool(enabled)
