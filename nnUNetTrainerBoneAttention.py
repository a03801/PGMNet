from __future__ import annotations

from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

try:
    from dynamic_network_architectures.architectures.BoneAttentionUNetV2 import BoneAttentionUNetForNNUNet
except ImportError:
    from BoneAttentionUNetV2 import BoneAttentionUNetForNNUNet


FEATURE_WIDTHS = (32, 64, 128, 256, 320, 320)
STRIDES = (
    (1, 1, 1),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 2),
)


def _to_class_map(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 4:
        return target.long()
    if target.ndim == 5:
        if target.shape[1] == 1:
            return target[:, 0].long()
        return target.argmax(dim=1).long()
    raise ValueError(f"Unsupported target shape: {tuple(target.shape)}")


def _resize_class_map(target: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
    if tuple(target.shape[-3:]) == tuple(size):
        return target
    return F.interpolate(target[:, None].float(), size=size, mode="nearest")[:, 0].long()


def _foreground_soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    num_classes = probabilities.shape[1]
    one_hot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    intersection = torch.sum(probabilities * one_hot, dim=dims)
    denominator = torch.sum(probabilities + one_hot, dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    if num_classes > 1:
        dice = dice[1:]
    return 1.0 - dice.mean()


class DynamicMultiscaleLoss(nn.Module):
    def __init__(
        self,
        trainer: "nnUNetTrainerBoneAttention",
        initial_weights: Tuple[float, float] = (0.3, 0.7),
        final_weights: Tuple[float, float] = (0.5, 0.5),
        warmup_epochs: int = 50,
    ):
        super().__init__()
        self.trainer = trainer
        self.initial_ce, self.initial_dice = map(float, initial_weights)
        self.final_ce, self.final_dice = map(float, final_weights)
        self.warmup_epochs = int(warmup_epochs)

    def _compound_weights(self) -> Tuple[float, float]:
        progress = min(1.0, float(self.trainer.current_epoch) / max(1, self.warmup_epochs))
        ce = self.initial_ce + progress * (self.final_ce - self.initial_ce)
        dice = self.initial_dice + progress * (self.final_dice - self.initial_dice)
        total = ce + dice
        return ce / total, dice / total

    def forward(self, outputs, targets):
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        if isinstance(targets, torch.Tensor):
            targets = [targets]

        full_target = _to_class_map(targets[0])
        scale_weights = np.array([0.5 ** i for i in range(len(outputs))], dtype=np.float64)
        scale_weights /= scale_weights.sum()

        ce_total = outputs[0].new_tensor(0.0)
        dice_total = outputs[0].new_tensor(0.0)
        for output, scale_weight in zip(outputs, scale_weights):
            target = _resize_class_map(full_target, tuple(output.shape[-3:]))
            ce_total = ce_total + float(scale_weight) * F.cross_entropy(output, target)
            dice_total = dice_total + float(scale_weight) * _foreground_soft_dice_loss(output, target)

        ce_weight, dice_weight = self._compound_weights()
        return ce_weight * ce_total + dice_weight * dice_total


def _ct_statistics_from_fingerprint(fingerprint: dict) -> Tuple[float, float]:
    properties = fingerprint.get("foreground_intensity_properties_per_channel")
    if not isinstance(properties, dict):
        raise RuntimeError("CT intensity statistics were not found in dataset_fingerprint.json")
    channel = properties.get("0", properties.get(0))
    if not isinstance(channel, dict) or "mean" not in channel or "std" not in channel:
        raise RuntimeError("Channel-0 CT mean/std were not found in dataset_fingerprint.json")
    mean = float(channel["mean"])
    std = float(channel["std"])
    if std <= 0:
        raise RuntimeError("The CT standard deviation in dataset_fingerprint.json must be positive")
    return mean, std


class nnUNetTrainerBoneAttention(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            unpack_dataset=unpack_dataset,
            device=device,
        )
        self.initial_lr = 0.01
        self.weight_decay = 3e-5
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.oversample_foreground_percent = 0.33
        self.enable_deep_supervision = True
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        patch_size = tuple(int(v) for v in self.configuration_manager.patch_size)
        spacing = tuple(float(v) for v in self.configuration_manager.spacing)
        batch_size = int(self.configuration_manager.batch_size)
        if patch_size != (96, 96, 96):
            raise RuntimeError(f"Expected a 96x96x96 3d_fullres patch, got {patch_size}")
        if batch_size != 2:
            raise RuntimeError(f"Expected batch size 2, got {batch_size}")
        if not np.allclose(spacing, (1.0, 1.0, 1.0), atol=1e-6):
            raise RuntimeError(f"Expected isotropic 1.0-mm spacing, got {spacing}")

    def _do_i_compile(self):
        return False

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if int(num_input_channels) != 1:
            raise RuntimeError(f"PGMNet expects one CT input channel, got {num_input_channels}")
        if int(num_output_channels) != 2:
            raise RuntimeError(f"PGMNet expects two segmentation outputs (background/lesion), got {num_output_channels}")
        return BoneAttentionUNetForNNUNet(
            in_channels=1,
            num_classes=2,
            feature_map_sizes=FEATURE_WIDTHS,
            strides=STRIDES,
            deep_supervision=bool(enable_deep_supervision),
            ct_mean=0.0,
            ct_std=1.0,
        )

    def initialize(self):
        super().initialize()
        fingerprint_path = join(self.preprocessed_dataset_folder_base, "dataset_fingerprint.json")
        fingerprint = load_json(fingerprint_path)
        ct_mean, ct_std = _ct_statistics_from_fingerprint(fingerprint)

        network = self.network.module if self.is_ddp else self.network
        network.apgm.set_ct_normalization_statistics(ct_mean, ct_std)

    def _build_loss(self):
        return DynamicMultiscaleLoss(
            trainer=self,
            initial_weights=(0.3, 0.7),
            final_weights=(0.5, 0.5),
            warmup_epochs=50,
        )
