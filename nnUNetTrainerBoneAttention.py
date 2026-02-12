import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

# Ensure compile off
os.environ.setdefault("NNUNET_COMPILE", "0")
os.environ.setdefault("nnUNet_compile", "0")
os.environ.setdefault("MPLBACKEND", "Agg")
# Defaults consistent with nnU-Net
os.environ.setdefault("BONE_USE_DYNLOSS", "1")
os.environ.setdefault("BONE_USE_DEEP_SUPERVISION", "1")

# New: normalization & priors env (defaults conservative; you can enable as needed)
# HU normalization inside model (clip to [-200,1800] then -> [0,1])
os.environ.setdefault("BONE_USE_INMODEL_NORM", "0")
os.environ.setdefault("BONE_NORM_HU_MIN", "-200")
os.environ.setdefault("BONE_NORM_HU_MAX", "1800")

# Priors
os.environ.setdefault("BONE_USE_BONE_PRIOR", "1")
os.environ.setdefault("BONE_USE_ARTIFACT_PRIOR", "1")
os.environ.setdefault("BONE_ATT_ALPHA", "0.5")
os.environ.setdefault("BONE_ATT_LOC", "all")  # enc | dec | all
os.environ.setdefault("BONE_ATT_KERNEL", "3")

# Bone thresholds
os.environ.setdefault("BONE_BONE_HU_LOW", "180")      # fixed low
os.environ.setdefault("BONE_BONE_HU_HIGH", "1500")    # used if dynamic disabled

# Dynamic high switch and guard
os.environ.setdefault("BONE_PRIOR_DYNAMIC_HIGH", "1")     # enable dynamic high by default per your request
os.environ.setdefault("BONE_PRIOR_P_HIGH", "0.995")
os.environ.setdefault("BONE_PRIOR_HIGH_MIN_HU", "1200")
os.environ.setdefault("BONE_PRIOR_HIGH_MAX_HU", "1800")

# Optional heads
os.environ.setdefault("BONE_USE_CBAM", "1")
os.environ.setdefault("BONE_USE_ASPP", "1")
os.environ.setdefault("BONE_USE_FPN", "1")

# ---------------------------
# Label utilities
# ---------------------------
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
        sizes = [(ti.shape[-3] * ti.shape[-2] * ti.shape[-1]) for ti in target]
        idx = int(torch.tensor(sizes).argmax().item())
        return to_class_map(target[idx]), target, idx
    raise TypeError(f"Unsupported target type {type(target)}")

def resize_class_map(class_map: torch.Tensor, size):
    if class_map.shape[1:] == size:
        return class_map
    return F.interpolate(class_map.unsqueeze(1).float(), size=size, mode='nearest')[:, 0].long()

def soft_dice_loss_from_class(logits: torch.Tensor,
                              class_map: torch.Tensor,
                              smooth: float = 1e-5,
                              ignore_background: bool = True) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    C = probs.shape[1]
    one_hot = torch.nn.functional.one_hot(class_map.long(), num_classes=C).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = torch.sum(probs * one_hot, dim=dims)
    denom = torch.sum(probs + one_hot, dim=dims)
    dice_pc = (2 * inter + smooth) / (denom + smooth)
    if ignore_background and C > 1:
        dice_pc = dice_pc[1:]
    return 1 - dice_pc.mean()

# ---------------------------
# Dynamic multi-scale loss
# ---------------------------
class CustomDynMultiScaleLoss(nn.Module):
    def __init__(self, trainer_ref,
                 init_ce_w=0.3, init_dc_w=0.7,
                 target_ce_w=0.5, target_dc_w=0.5,
                 warmup_epochs=50):
        super().__init__()
        self.trainer = trainer_ref
        self.init_ce_w = init_ce_w
        self.init_dc_w = init_dc_w
        self.target_ce_w = target_ce_w
        self.target_dc_w = target_dc_w
        self.warmup_epochs = warmup_epochs
        self.ce_w = init_ce_w
        self.dc_w = init_dc_w
        self.step_counter = 0
        self.first_logged = False

    def _epoch(self):
        return getattr(self.trainer, "current_epoch", 0)

    def _update_weights(self):
        prog = min(1.0, self._epoch() / max(1, self.warmup_epochs))
        new_ce = self.init_ce_w + (self.target_ce_w - self.init_ce_w) * prog
        new_dc = self.init_dc_w + (self.target_dc_w - self.init_dc_w) * prog
        s = new_ce + new_dc
        self.ce_w, self.dc_w = new_ce / s, new_dc / s

    def forward(self, outputs, target):
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        full_res_map, target_list, full_idx = select_full_res_target(target)

        raw_w = [0.5 ** i for i in range(len(outputs))]
        sw = sum(raw_w)
        weights = [w / sw for w in raw_w]

        device = outputs[0].device
        total_ce = torch.tensor(0.0, device=device)
        total_dice = torch.tensor(0.0, device=device)

        for i, (logit, w) in enumerate(zip(outputs, weights)):
            scaled = resize_class_map(full_res_map, logit.shape[2:])
            ce_i = F.cross_entropy(logit, scaled)
            dice_i = soft_dice_loss_from_class(logit, scaled, ignore_background=True)
            total_ce += w * ce_i
            total_dice += w * dice_i
            if not self.first_logged:
                print(f"[DynLoss] out{i} logit={tuple(logit.shape)} tgt={tuple(scaled.shape)} "
                      f"w={w:.3f} ce={ce_i.item():.4f} dice={dice_i.item():.4f}")

        self._update_weights()
        total = self.ce_w * total_ce + self.dc_w * total_dice

        if not self.first_logged:
            if target_list is not None:
                shapes = [tuple(t.shape) for t in target_list]
                print(f"[DynLoss] TargetList len={len(target_list)} full_idx={full_idx} shapes={shapes} "
                      f"full_res_class={tuple(full_res_map.shape)}")
            else:
                print(f"[DynLoss] Single target full_res_class={tuple(full_res_map.shape)}")
            print(f"[DynLoss] step={self.step_counter} epoch={self._epoch()} "
                  f"ce_w={self.ce_w:.3f} dc_w={self.dc_w:.3f} "
                  f"ce={total_ce.item():.4f} dice={total_dice.item():.4f} total={total.item():.4f}")
            self.first_logged = True
        else:
            if self.step_counter < 10 or self.step_counter % 50 == 0:
                print(f"[DynLoss] step={self.step_counter} epoch={self._epoch()} "
                      f"ce_w={self.ce_w:.3f} dc_w={self.dc_w:.3f} "
                      f"ce={total_ce.item():.4f} dice={total_dice.item():.4f} total={total.item():.4f}")
        self.step_counter += 1
        return total

# ---------------------------
# Helpers
# ---------------------------
def _env_flag(name: str) -> bool:
    return os.environ.get(name, '0').lower() in ('1', 'true', 't', 'yes', 'y')

def _num_classes_from_dataset_json(dataset_json) -> int:
    try:
        labels = dataset_json.get("labels", {})
        if isinstance(labels, dict) and len(labels):
            mx = max(int(v) for v in labels.values())
            return int(mx + 1)
    except Exception:
        pass
    return 2

def _safe_import_bone_net():
    try:
        from dynamic_network_architectures.architectures.BoneAttentionUNetV2 import BoneAttentionUNetForNNUNet
        return BoneAttentionUNetForNNUNet
    except Exception:
        try:
            from BoneAttentionUNetV2 import BoneAttentionUNetForNNUNet
            return BoneAttentionUNetForNNUNet
        except Exception as e:
            raise ImportError(
                "Cannot import BoneAttentionUNetForNNUNet. "
                "Ensure BoneAttentionUNetV2.py is importable either as "
                "dynamic_network_architectures.architectures.BoneAttentionUNetV2 "
                "or in PYTHONPATH."
            ) from e

def _default_fms_and_strides():
    feature_map_sizes = (32, 64, 128, 256, 320)
    # feature_map_sizes = (32, 64, 128, 256)
    strides = ((1,1,1),(2,2,2),(2,2,2),(2,2,2),(2,2,2))
    return feature_map_sizes, strides

class _TopOnlyLoss(nn.Module):
    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, net_output, target):
        if isinstance(net_output, list):
            net_output = net_output[0]
        class_map, _, _ = select_full_res_target(target)
        if class_map.shape[1:] != tuple(net_output.shape[2:]):
            class_map = F.interpolate(
                class_map.unsqueeze(1).float(),
                size=net_output.shape[2:],
                mode='nearest'
            )[:, 0].long()
        target_for_dc_ce = class_map.unsqueeze(1).long()
        return self.base_loss(net_output, target_for_dc_ce)

# ---------------------------
# Trainer
# ---------------------------
class nnUNetTrainerBoneAttention(nnUNetTrainer):
    """
    Env:
      BONE_USE_CBAM, BONE_USE_ASPP, BONE_USE_FPN
      BONE_USE_DYNLOSS, BONE_USE_DEEP_SUPERVISION
      BONE_USE_INMODEL_NORM, BONE_NORM_HU_MIN, BONE_NORM_HU_MAX
      BONE_USE_BONE_PRIOR, BONE_USE_ARTIFACT_PRIOR
      BONE_ATT_ALPHA, BONE_ATT_LOC, BONE_ATT_KERNEL
      BONE_BONE_HU_LOW, BONE_BONE_HU_HIGH
      BONE_PRIOR_DYNAMIC_HIGH, BONE_PRIOR_P_HIGH, BONE_PRIOR_HIGH_MIN_HU, BONE_PRIOR_HIGH_MAX_HU
    """
    def __init__(self, plans, configuration, fold, dataset_json,
                 device: torch.device = torch.device('cuda')):
        print("[BoneTrainer] __init__ start")
        super().__init__(plans, configuration, fold, dataset_json, device=device)

        self.use_cbam = _env_flag("BONE_USE_CBAM")
        self.use_aspp = _env_flag("BONE_USE_ASPP")
        self.use_fpn  = _env_flag("BONE_USE_FPN")

        # new flags/params
        self.use_inmodel_norm = _env_flag("BONE_USE_INMODEL_NORM")
        self.norm_hu_min = float(os.environ.get("BONE_NORM_HU_MIN", "-200"))
        self.norm_hu_max = float(os.environ.get("BONE_NORM_HU_MAX", "1800"))

        self.use_bone_prior = _env_flag("BONE_USE_BONE_PRIOR")
        self.use_artifact_prior = _env_flag("BONE_USE_ARTIFACT_PRIOR")
        self.prior_alpha = float(os.environ.get("BONE_ATT_ALPHA", "0.5"))
        self.prior_loc = os.environ.get("BONE_ATT_LOC", "all")
        self.prior_kernel = int(os.environ.get("BONE_ATT_KERNEL", "3"))

        self.bone_hu_low = float(os.environ.get("BONE_BONE_HU_LOW", "180"))
        self.bone_hu_high = float(os.environ.get("BONE_BONE_HU_HIGH", "1500"))

        self.use_dynamic_high = _env_flag("BONE_PRIOR_DYNAMIC_HIGH")
        self.dynamic_high_p = float(os.environ.get("BONE_PRIOR_P_HIGH", "0.995"))
        self.dynamic_high_min_hu = float(os.environ.get("BONE_PRIOR_HIGH_MIN_HU", "1200"))
        self.dynamic_high_max_hu = float(os.environ.get("BONE_PRIOR_HIGH_MAX_HU", "1800"))

        self.num_input_channels = None
        self.num_classes = None

        self.disable_validation = False
        self.perform_validation_every_x_epochs = 1

        self.last_val_loss = None
        self.last_dice_per_class = None
        self.last_mean_fg = None
        self.logged_memory_once = False

        print(f"[BoneTrainer] CBAM={self.use_cbam} ASPP={self.use_aspp} FPN={self.use_fpn}")
        print(f"[BoneTrainer] Norm(in-model)={self.use_inmodel_norm} HU=[{self.norm_hu_min},{self.norm_hu_max}]")
        print(f"[BoneTrainer] Priors Bone={self.use_bone_prior} Artifact={self.use_artifact_prior} "
              f"alpha={self.prior_alpha} loc={self.prior_loc} k={self.prior_kernel}")
        print(f"[BoneTrainer] Bone low={self.bone_hu_low} high(fixed)={self.bone_hu_high} "
              f"DynamicHigh={self.use_dynamic_high} p={self.dynamic_high_p} "
              f"Guard=[{self.dynamic_high_min_hu},{self.dynamic_high_max_hu}]")
        print("[BoneTrainer] __init__ done; waiting for network build")

    @staticmethod
    def build_network_architecture(*args, **kwargs):
        self_ref = None
        if len(args) > 0 and isinstance(args[0], nnUNetTrainerBoneAttention):
            self_ref = args[0]
            args = args[1:]

        net_in_ch = None
        net_num_classes = None
        enable_ds = True
        feature_map_sizes, strides = _default_fms_and_strides()

        if len(args) >= 6:
            arch_name, arch_conf, kw_list, net_in_ch, net_num_classes, enable_ds = args[:6]
            if isinstance(arch_conf, dict):
                fms = arch_conf.get("features_per_stage") or arch_conf.get("num_features_per_stage")
                if isinstance(fms, (list, tuple)) and len(fms) >= 5:
                    feature_map_sizes = tuple(int(x) for x in fms[:5])
                s_plan = arch_conf.get("strides")
                if isinstance(s_plan, (list, tuple)) and len(s_plan) >= len(feature_map_sizes):
                    strides = tuple(tuple(int(x) for x in s) for s in s_plan[:len(feature_map_sizes)])
        elif len(args) >= 5:
            plans_manager, dataset_json, configuration_manager, net_in_ch, enable_ds = args[:5]
            net_num_classes = _num_classes_from_dataset_json(dataset_json)
            try:
                if hasattr(configuration_manager, "pool_op_kernel_sizes"):
                    s_plan = configuration_manager.pool_op_kernel_sizes
                    if isinstance(s_plan, (list, tuple)) and len(s_plan) >= 1:
                        strides = tuple(tuple(int(x) for x in s) for s in s_plan[:5])
                if hasattr(configuration_manager, "num_channels_per_stage"):
                    fms = configuration_manager.num_channels_per_stage
                    if isinstance(fms, (list, tuple)) and len(fms) >= 5:
                        feature_map_sizes = tuple(int(x) for x in fms[:5])
            except Exception:
                pass
        else:
            raise RuntimeError("nnUNetTrainerBoneAttention.build_network_architecture: unsupported call signature.")

        use_deep_sup_env = _env_flag("BONE_USE_DEEP_SUPERVISION")
        enable_ds = bool(use_deep_sup_env)

        if self_ref is not None and hasattr(self_ref, "device"):
            device = self_ref.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        use_cbam = getattr(self_ref, "use_cbam", _env_flag("BONE_USE_CBAM"))
        use_aspp = getattr(self_ref, "use_aspp", _env_flag("BONE_USE_ASPP"))
        use_fpn  = getattr(self_ref, "use_fpn",  _env_flag("BONE_USE_FPN"))

        use_inmodel_norm = getattr(self_ref, "use_inmodel_norm", _env_flag("BONE_USE_INMODEL_NORM"))
        norm_hu_min = getattr(self_ref, "norm_hu_min", float(os.environ.get("BONE_NORM_HU_MIN", "-200")))
        norm_hu_max = getattr(self_ref, "norm_hu_max", float(os.environ.get("BONE_NORM_HU_MAX", "1800")))

        use_bone_prior = getattr(self_ref, "use_bone_prior", _env_flag("BONE_USE_BONE_PRIOR"))
        use_artifact_prior = getattr(self_ref, "use_artifact_prior", _env_flag("BONE_USE_ARTIFACT_PRIOR"))
        prior_alpha = getattr(self_ref, "prior_alpha", float(os.environ.get("BONE_ATT_ALPHA", "0.5")))
        prior_apply_to = getattr(self_ref, "prior_loc", os.environ.get("BONE_ATT_LOC", "all"))
        prior_kernel_size = getattr(self_ref, "prior_kernel", int(os.environ.get("BONE_ATT_KERNEL", "3")))

        bone_hu_low = getattr(self_ref, "bone_hu_low", float(os.environ.get("BONE_BONE_HU_LOW", "180")))
        bone_hu_high = getattr(self_ref, "bone_hu_high", float(os.environ.get("BONE_BONE_HU_HIGH", "1500")))

        use_dynamic_high = getattr(self_ref, "use_dynamic_high", _env_flag("BONE_PRIOR_DYNAMIC_HIGH"))
        dynamic_high_p = getattr(self_ref, "dynamic_high_p", float(os.environ.get("BONE_PRIOR_P_HIGH", "0.995")))
        dynamic_high_min_hu = getattr(self_ref, "dynamic_high_min_hu", float(os.environ.get("BONE_PRIOR_HIGH_MIN_HU", "1200")))
        dynamic_high_max_hu = getattr(self_ref, "dynamic_high_max_hu", float(os.environ.get("BONE_PRIOR_HIGH_MAX_HU", "1800")))

        BoneAttentionUNetForNNUNet = _safe_import_bone_net()

        net = BoneAttentionUNetForNNUNet(
            in_channels=net_in_ch,
            num_classes=_num_classes_from_dataset_json(args[1]) if len(args) >= 5 else net_num_classes,
            feature_map_sizes=feature_map_sizes,
            strides=strides,
            use_cbam=use_cbam,
            use_aspp=use_aspp,
            use_fpn=use_fpn,
            deep_supervision=enable_ds,
            max_ds_outputs=4,
            # new
            use_inmodel_norm=use_inmodel_norm,
            norm_hu_min=norm_hu_min,
            norm_hu_max=norm_hu_max,
            use_bone_prior=use_bone_prior,
            use_artifact_prior=use_artifact_prior,
            prior_alpha=prior_alpha,
            prior_apply_to=prior_apply_to,
            prior_kernel_size=prior_kernel_size,
            bone_hu_low=bone_hu_low,
            bone_hu_high=bone_hu_high,
            use_dynamic_high=use_dynamic_high,
            dynamic_high_p=dynamic_high_p,
            dynamic_high_min_hu=dynamic_high_min_hu,
            dynamic_high_max_hu=dynamic_high_max_hu
        ).to(device)

        if not hasattr(net, "decoder"):
            class _DecProxy(nn.Module):
                def __init__(self, flag): super().__init__(); self.deep_supervision = flag
            net.decoder = _DecProxy(enable_ds)

        if self_ref is not None:
            self_ref.network = net
            self_ref.num_input_channels = net_in_ch
            self_ref.num_classes = _num_classes_from_dataset_json(args[1]) if len(args) >= 5 else net_num_classes
            self_ref.enable_deep_supervision = enable_ds
            params = sum(p.numel() for p in net.parameters())
            print(f"[BoneTrainer] Built BoneAttentionNet params={params/1e6:.2f}M "
                  f"FMS={feature_map_sizes} strides={strides} cbam={use_cbam} aspp={use_aspp} fpn={use_fpn} "
                  f"deep_supervision={enable_ds}")
        else:
            params = sum(p.numel() for p in net.parameters())
            print(f"[BoneTrainer] Built BoneAttentionNet (infer) params={params/1e6:.2f}M")

        return net

    def on_train_start(self):
        super().on_train_start()
        use_deep_sup = _env_flag("BONE_USE_DEEP_SUPERVISION")
        self.set_deep_supervision_enabled(use_deep_sup)

        if _env_flag("BONE_USE_DYNLOSS"):
            self.loss = CustomDynMultiScaleLoss(
                trainer_ref=self,
                init_ce_w=0.3, init_dc_w=0.7,
                target_ce_w=0.5, target_dc_w=0.5,
                warmup_epochs=50
            )
            print("[BoneTrainer] Using CustomDynMultiScaleLoss.")
        else:
            from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
            base_loss = DC_and_CE_loss(
                soft_dice_kwargs={"batch_dice": False, "smooth": 1e-5, "do_bg": False},
                ce_kwargs={}
            )
            self.loss = _TopOnlyLoss(base_loss)
            print("[BoneTrainer] Using default DC+CE loss (top only).")

        log = self.logger.my_fantastic_logging
        log['modules_used'] = {
            'cbam': self.use_cbam,
            'aspp': self.use_aspp,
            'fpn': self.use_fpn,
            'dynloss': _env_flag("BONE_USE_DYNLOSS"),
            'deep_supervision': use_deep_sup,
            'inmodel_norm': self.use_inmodel_norm,
            'norm_hu_min': self.norm_hu_min,
            'norm_hu_max': self.norm_hu_max,
            'bone_prior': self.use_bone_prior,
            'artifact_prior': self.use_artifact_prior,
            'att_alpha': self.prior_alpha,
            'att_loc': self.prior_loc,
            'att_kernel': self.prior_kernel,
            'bone_hu_low': self.bone_hu_low,
            'bone_hu_high': self.bone_hu_high,
            'prior_dynamic_high': self.use_dynamic_high,
            'prior_p_high': self.dynamic_high_p,
            'prior_high_guard_min': self.dynamic_high_min_hu,
            'prior_high_guard_max': self.dynamic_high_max_hu
        }
        print(f"[BoneTrainer] Modules used -> {log['modules_used']}")

    def should_validate_now(self):
        if self.disable_validation:
            return False
        epoch = getattr(self, "current_epoch", 0)
        if (epoch + 1) % self.perform_validation_every_x_epochs != 0:
            return False
        return super().should_validate_now()

    def train_step(self, batch: dict):
        out = super().train_step(batch)
        if not self.logged_memory_once:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                print(f"[BoneTrainer] Peak GPU memory after first train_step: {mem:.2f} MB")
            self.logged_memory_once = True
        return out

    def validation_step(self, batch: dict):
        if self.disable_validation:
            return None
        self.network.eval()
        data = batch['data']
        target = batch['target']
        if isinstance(data, (list, tuple)):
            raise RuntimeError("Unexpected list/tuple data in validation_step.")
        data = data.to(self.device, non_blocking=True, dtype=torch.float32)

        with torch.no_grad():
            out = self.network(data)
            if isinstance(out, list):
                out = out[0]
            probs = torch.softmax(out, dim=1)
            pred = torch.argmax(probs, dim=1, keepdim=True)

        full_map, _, _ = select_full_res_target(target)
        gt = full_map.unsqueeze(1).to(pred.device)

        C = out.shape[1]
        pred_onehot = torch.zeros((pred.shape[0], C, *pred.shape[2:]),
                                  device=pred.device, dtype=torch.int)
        pred_onehot.scatter_(1, pred.long(), 1)

        axes = list(range(2, pred_onehot.ndim))
        tp, fp, fn, tn = get_tp_fp_fn_tn(pred_onehot, gt.long(), axes=axes, mask=None)
        return {
            'tp': tp.sum(0),
            'fp': fp.sum(0),
            'fn': fn.sum(0),
            'tn': tn.sum(0),
            'num_samples': pred.shape[0]
        }

    def on_validation_epoch_end(self, val_outputs):
        if self.disable_validation:
            return
        if not val_outputs:
            print("[Val] WARNING: empty val_outputs")
            return
        val_outputs = [v for v in val_outputs if v is not None]
        if len(val_outputs) == 0:
            print("[Val] WARNING: only None val outputs")
            return

        tp = torch.stack([v['tp'] for v in val_outputs], 0).sum(0)
        fp = torch.stack([v['fp'] for v in val_outputs], 0).sum(0)
        fn = torch.stack([v['fn'] for v in val_outputs], 0).sum(0)
        eps = 1e-8
        dice_per_class = (2 * tp + eps) / (2 * tp + fp + fn + eps)

        if dice_per_class.numel() > 1:
            mean_fg = dice_per_class[1:].mean().item()
        else:
            mean_fg = dice_per_class.mean().item()

        val_loss_proxy = 1 - mean_fg
        self.last_val_loss = val_loss_proxy
        self.last_dice_per_class = dice_per_class.detach().cpu().numpy()
        self.last_mean_fg = mean_fg

        log = self.logger.my_fantastic_logging
        for k in ('val_losses','dice_per_class_or_region','ema_fg_dice','fg_dice'):
            if k not in log: log[k] = []
        if len(log['ema_fg_dice']) == 0:
            ema = mean_fg
        else:
            ema = 0.9 * log['ema_fg_dice'][-1] + 0.1 * mean_fg

        log['val_losses'].append(val_loss_proxy)
        log['dice_per_class_or_region'].append(self.last_dice_per_class)
        log['fg_dice'].append(mean_fg)
        log['ema_fg_dice'].append(ema)

        epoch = getattr(self, "current_epoch", 0)
        dstr = ", ".join([f"{i}:{d.item():.4f}" for i, d in enumerate(dice_per_class)])
        print(f"[Val] epoch={epoch} DicePerClass[{dstr}] MeanDice(fg)={mean_fg:.4f} EMA={ema:.4f} val_loss_proxy={val_loss_proxy:.4f}")

    def set_deep_supervision_enabled(self, enabled: bool):
        self.enable_deep_supervision = enabled
        applied = False
        if hasattr(self, "network"):
            if hasattr(self.network, "decoder") and hasattr(self.network.decoder, "deep_supervision"):
                self.network.decoder.deep_supervision = enabled; applied = True
            if hasattr(self.network, "deep_supervision"):
                try:
                    self.network.deep_supervision = enabled; applied = True
                except Exception:
                    pass
        print(f"[BoneTrainer] set_deep_supervision_enabled -> {enabled} (applied={applied})")
