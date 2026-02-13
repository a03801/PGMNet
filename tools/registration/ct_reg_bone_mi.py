#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CT-only registration (Rigid->Affine) with MI computed on bone ROI (SimpleITK).

No private paths are hard-coded (safe for GitHub).
You must provide either:
  A) --pre_ct and --post_ct (full paths), plus --out_dir
or
  B) --case_id (first 3 chars) with --pre_base/--post_base, plus --out_base (or --out_dir)

Pipeline:
- Resample fixed/moving to isotropic spacing
- Build bone ROI masks on normalized CT in [0,1]
    bone = (img > BONE_THR) AND (img <= BONE_HI)
  where BONE_HI is HARD-CODED to 0.90 (exclude metal/saturation)
- Optional shape-based rigid init using distance maps of bone masks (MeanSquares)
- Rigid MI on bone ROI
- Affine MI on bone ROI
- Resample moving into fixed space
- QC: NCC within fixed bone ROI + optional QC PNG montage

Outputs (out_dir):
- mask_init_rigid.tfm            (if enabled and masks sufficiently large)
- final_transform.tfm
- registered_postop_ct.nii.gz
- fixed_bone_mask.nii.gz
- moving_bone_mask.nii.gz
- qc_preview.png                 (if --qc_png)
- report.json

Assumption:
- CT intensities normalized to [0,1] (recommended). Output is float32 and clamped to [0,1].
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as e:
    raise SystemExit("Missing dependency: SimpleITK. Install via `pip install SimpleITK`.") from e


# ==========================
# HARD-CODED bone mask params
# ==========================
BONE_THR = 0.19
BONE_HI  = 0.90   # <- fixed: values > 0.90 are excluded from bone ROI


# ==========================
# Auto find helpers
# ==========================
def norm_case_id(case_id: str, n: int = 3) -> str:
    s = str(case_id).strip()
    if s.isdigit():
        s = s.zfill(n)
    return s[:n]


def list_candidates(base: Path, prefix3: str) -> List[Path]:
    if not base.exists():
        return []
    pat = f"{prefix3}*.nii*"
    c1 = sorted(base.glob(pat))
    if c1:
        return c1
    try:
        return sorted(base.rglob(pat))
    except Exception:
        return []


def score_pre_ct(p: Path) -> float:
    name = p.name.lower()
    score = 0.0
    if "_0000" in name:
        score += 50.0
    if any(k in name for k in ["mask", "seg", "label", "resection", "cavity"]):
        score -= 200.0
    try:
        score += min(p.stat().st_size / (1024 * 1024), 200.0) * 0.1
    except Exception:
        pass
    return score


def score_post_ct(p: Path) -> float:
    name = p.name.lower()
    score = 0.0
    if any(k in name for k in ["mask", "seg", "label", "resection", "cavity"]):
        score -= 200.0
    else:
        score += 30.0
    try:
        score += min(p.stat().st_size / (1024 * 1024), 200.0) * 0.1
    except Exception:
        pass
    return score


def pick_best(cands: List[Path], scorer) -> Tuple[Optional[Path], List[Tuple[Path, float]]]:
    scored: List[Tuple[Path, float]] = []
    for p in cands:
        try:
            s = float(scorer(p))
        except Exception:
            s = -1e9
        scored.append((p, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0][0] if scored else None
    return best, scored


def auto_find_paths(prefix3: str, pre_base: Path, post_base: Path) -> Tuple[Optional[Path], Optional[Path]]:
    pre_cands = list_candidates(pre_base, prefix3)
    pre_best, pre_scored = pick_best(pre_cands, score_pre_ct)

    post_cands = list_candidates(post_base, prefix3)
    post_best, post_scored = pick_best(post_cands, score_post_ct)

    print(f"[AUTO] case prefix = {prefix3}")
    print(f"[AUTO] pre_base  = {pre_base}")
    print(f"[AUTO] post_base = {post_base}")

    def _print_top(title: str, scored_list: List[Tuple[Path, float]]):
        print(f"[AUTO] {title} candidates (top 8):")
        for p, s in scored_list[:8]:
            print(f"       {s:8.2f}  {p}")

    _print_top("PRE_CT", pre_scored)
    _print_top("POST_CT", post_scored)

    print(f"[AUTO] Picked PRE_CT  : {pre_best}")
    print(f"[AUTO] Picked POST_CT : {post_best}")
    return pre_best, post_best


def require_path(p: Optional[Path], label: str) -> Path:
    if p is None:
        raise ValueError(f"Missing required path: {label}")
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"[{label}] not found: {p}")
    return p


# ==========================
# SimpleITK helpers
# ==========================
def read_image(path: Path, pixel_type=sitk.sitkFloat32) -> sitk.Image:
    img = sitk.ReadImage(str(path))
    return sitk.Cast(img, pixel_type)


def resample_isotropic(img: sitk.Image, out_spacing=(1.0, 1.0, 1.0),
                       is_label: bool = False, default_value: float = 0.0) -> sitk.Image:
    in_spacing = img.GetSpacing()
    in_size = img.GetSize()
    out_size = [int(np.round(in_size[i] * in_spacing[i] / out_spacing[i])) for i in range(3)]

    res = sitk.ResampleImageFilter()
    res.SetOutputSpacing(out_spacing)
    res.SetSize(out_size)
    res.SetOutputDirection(img.GetDirection())
    res.SetOutputOrigin(img.GetOrigin())
    res.SetTransform(sitk.Transform())
    res.SetDefaultPixelValue(float(default_value))
    res.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return res.Execute(img)


def to_uint8_bone_mask(img: sitk.Image) -> sitk.Image:
    mask = sitk.And(sitk.Greater(img, float(BONE_THR)),
                    sitk.LessEqual(img, float(BONE_HI)))
    return sitk.Cast(mask, sitk.sitkUInt8)


def mask_voxel_count(mask_u8: sitk.Image) -> int:
    st = sitk.StatisticsImageFilter()
    st.Execute(mask_u8)
    return int(round(st.GetSum()))


def image_ncc(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    m = np.ones_like(a, dtype=bool) if mask is None else mask.astype(bool)

    av = a[m].ravel()
    bv = b[m].ravel()
    if av.size < 2000:
        return float("nan")

    av = av - av.mean()
    bv = bv - bv.mean()
    denom = (np.sqrt((av * av).sum()) * np.sqrt((bv * bv).sum())) + 1e-8
    return float((av * bv).sum() / denom)


def _unwrap_last_transform(t: sitk.Transform) -> sitk.Transform:
    try:
        if t.GetName() == "CompositeTransform":
            ct = sitk.CompositeTransform(t)
            n = ct.GetNumberOfTransforms()
            if n > 0:
                return _unwrap_last_transform(ct.GetNthTransform(n - 1))
    except Exception:
        pass
    return t


# ==========================
# Registration configs
# ==========================
def setup_mi_reg(metric_bins: int, sampling_pct: float, iters: int, lr: float,
                 shrink: Tuple[int, ...], smooth: Tuple[int, ...]) -> sitk.ImageRegistrationMethod:
    if len(shrink) != len(smooth):
        raise ValueError("shrink and smooth must have same length.")

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(metric_bins))
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(float(sampling_pct), seed=42)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsRegularStepGradientDescent(
        float(lr), 1e-4, int(iters), 0.5, 1e-8
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    reg.SetShrinkFactorsPerLevel(shrinkFactors=list(shrink))
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=list(smooth))
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return reg


def setup_mask_shape_reg(iters: int = 500, lr: float = 4.0,
                         shrink: Tuple[int, ...] = (8, 4, 2, 1),
                         smooth: Tuple[int, ...] = (3, 2, 1, 0)) -> sitk.ImageRegistrationMethod:
    if len(shrink) != len(smooth):
        raise ValueError("shrink and smooth must have same length.")

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMeanSquares()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(float(lr), 1e-4, int(iters), 0.5, 1e-8)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(shrinkFactors=list(shrink))
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=list(smooth))
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return reg


def rigid_init_from_bone_masks(fixed_bone: sitk.Image, moving_bone: sitk.Image) -> sitk.Euler3DTransform:
    fixed_dm = sitk.Abs(sitk.SignedMaurerDistanceMap(
        fixed_bone, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
    ))
    moving_dm = sitk.Abs(sitk.SignedMaurerDistanceMap(
        moving_bone, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
    ))
    fixed_dm = sitk.Cast(fixed_dm, sitk.sitkFloat32)
    moving_dm = sitk.Cast(moving_dm, sitk.sitkFloat32)

    init = sitk.CenteredTransformInitializer(
        fixed_dm, moving_dm,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    reg = setup_mask_shape_reg()
    reg.SetInitialTransform(init, inPlace=True)
    out = _unwrap_last_transform(reg.Execute(fixed_dm, moving_dm))

    if out.GetName() != "Euler3DTransform":
        try:
            out = sitk.Euler3DTransform(out)
        except Exception:
            out = init
    return out


def register_rigid_affine_mi(
    fixed_ct: sitk.Image,
    moving_ct: sitk.Image,
    fixed_bone: sitk.Image,
    moving_bone: sitk.Image,
    use_moving_mask: bool,
    enable_mask_init: bool,
    out_dir: Path
) -> sitk.Transform:
    # stage 0: optional mask-init
    if enable_mask_init:
        f_cnt = mask_voxel_count(fixed_bone)
        m_cnt = mask_voxel_count(moving_bone)
        if f_cnt > 1000 and m_cnt > 1000:
            print(f"[MASK-INIT] enabled. fixed bone voxels={f_cnt}, moving bone voxels={m_cnt}")
            init_rigid = rigid_init_from_bone_masks(fixed_bone, moving_bone)
            try:
                sitk.WriteTransform(init_rigid, str(out_dir / "mask_init_rigid.tfm"))
            except Exception:
                pass
        else:
            print(f"[MASK-INIT] skipped (bone mask too small). fixed={f_cnt}, moving={m_cnt}")
            init_rigid = sitk.CenteredTransformInitializer(
                fixed_ct, moving_ct, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
    else:
        init_rigid = sitk.CenteredTransformInitializer(
            fixed_ct, moving_ct, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

    # stage 1: rigid MI
    reg1 = setup_mi_reg(60, 0.5, 700, 2.0, (8, 4, 2, 1), (3, 2, 1, 0))
    reg1.SetMetricFixedMask(fixed_bone)
    if use_moving_mask:
        reg1.SetMetricMovingMask(moving_bone)
    reg1.SetInitialTransform(init_rigid, inPlace=True)
    rigid = _unwrap_last_transform(reg1.Execute(fixed_ct, moving_ct))

    if rigid.GetName() != "Euler3DTransform":
        try:
            rigid = sitk.Euler3DTransform(rigid)
        except Exception:
            pass

    # stage 2: affine MI
    init_aff = sitk.AffineTransform(3)
    init_aff.SetCenter(rigid.GetCenter())
    init_aff.SetMatrix(rigid.GetMatrix())
    init_aff.SetTranslation(rigid.GetTranslation())

    reg2 = setup_mi_reg(60, 0.35, 900, 1.0, (4, 2, 1), (2, 1, 0))
    reg2.SetMetricFixedMask(fixed_bone)
    if use_moving_mask:
        reg2.SetMetricMovingMask(moving_bone)
    reg2.SetInitialTransform(init_aff, inPlace=True)
    affine = _unwrap_last_transform(reg2.Execute(fixed_ct, moving_ct))
    return affine


# ==========================
# QC montage
# ==========================
def pick_best_slice(mask_zyx: np.ndarray) -> int:
    areas = mask_zyx.reshape(mask_zyx.shape[0], -1).sum(axis=1)
    if areas.max() <= 0:
        return mask_zyx.shape[0] // 2
    return int(np.argmax(areas))


def save_qc_png(out_dir: Path, fixed_arr: np.ndarray, moving_arr: np.ndarray, reg_arr: np.ndarray,
                fixed_bone: np.ndarray, save_png: bool, show: bool) -> Optional[Path]:
    if (not save_png) and (not show):
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib not available, skip QC images. ({e})")
        return None

    z = pick_best_slice(fixed_bone.astype(np.uint8))
    f = fixed_arr[z, :, :]
    m = moving_arr[z, :, :]
    r = reg_arr[z, :, :]
    d = np.abs(f - r)

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1); ax1.imshow(f, cmap="gray", vmin=0, vmax=1); ax1.set_title(f"Fixed (Pre) z={z}"); ax1.axis("off")
    ax2 = fig.add_subplot(2, 2, 2); ax2.imshow(m, cmap="gray", vmin=0, vmax=1); ax2.set_title("Moving (Post, iso)"); ax2.axis("off")
    ax3 = fig.add_subplot(2, 2, 3); ax3.imshow(r, cmap="gray", vmin=0, vmax=1); ax3.set_title("Registered Post"); ax3.axis("off")
    ax4 = fig.add_subplot(2, 2, 4); ax4.imshow(d, cmap="gray"); ax4.set_title("|Fixed - Registered|"); ax4.axis("off")

    qc_path = out_dir / "qc_preview.png"
    if save_png:
        fig.savefig(str(qc_path), dpi=150, bbox_inches="tight")
        print(f"[OK] QC image saved: {qc_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return qc_path if save_png else None


# ==========================
# CLI
# ==========================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="CT-only Rigid->Affine registration with MI on bone ROI (SimpleITK)."
    )

    p.add_argument("--pre_ct", type=Path, default=None, help="Preoperative CT (fixed), normalized to [0,1]")
    p.add_argument("--post_ct", type=Path, default=None, help="Postoperative CT (moving), normalized to [0,1]")
    p.add_argument("--case_id", type=str, default=None, help="Case id prefix (前三位，例如 005).")

    p.add_argument("--pre_base", type=Path, default=None, help="Base directory for pre-op CT inputs (required if using --case_id)")
    p.add_argument("--post_base", type=Path, default=None, help="Base directory for post-op CT inputs (required if using --case_id)")
    p.add_argument("--out_dir", type=Path, default=None, help="Output directory (recommended).")
    p.add_argument("--out_base", type=Path, default=None, help="Output base directory (used when --out_dir not set).")

    p.add_argument("--iso", type=float, default=1.0, help="Isotropic spacing (mm)")
    p.add_argument("--use_moving_mask", action="store_true", default=False, help="Also use moving bone mask for metric.")
    p.add_argument("--disable_mask_init", action="store_true", default=False, help="Disable mask-shape initialization.")
    p.add_argument("--qc_ncc_threshold", type=float, default=0.85, help="Warn if NCC < threshold.")
    p.add_argument("--qc_png", action="store_true", default=False, help="Save QC montage PNG.")
    p.add_argument("--display_images", action="store_true", default=False, help="Show QC montage via matplotlib.")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None and ("ipykernel" in sys.modules):
        if ("--case_id" not in sys.argv) and ("--pre_ct" not in sys.argv) and ("--post_ct" not in sys.argv):
            print(
                "[INFO] Jupyter detected, but no CLI args were provided.\n"
                "Run e.g.:\n"
                "  %run ct_reg_bone_mi.py --pre_ct ... --post_ct ... --out_dir ... --qc_png\n"
            )
            return 0

    args = build_parser().parse_args(argv)

    prefix3: Optional[str] = None
    if args.case_id:
        prefix3 = norm_case_id(args.case_id, 3)

    pre_path: Optional[Path] = None
    post_path: Optional[Path] = None

    # Mode A: explicit paths
    if args.pre_ct is not None and args.post_ct is not None:
        pre_path = require_path(args.pre_ct, "pre_ct")
        post_path = require_path(args.post_ct, "post_ct")
        if args.out_dir is None:
            raise ValueError("When using --pre_ct/--post_ct, please also provide --out_dir.")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    else:
        # Mode B: case_id + bases
        if prefix3 is None:
            raise ValueError("Provide either (--pre_ct and --post_ct) or --case_id.")
        pre_base = require_path(args.pre_base, "pre_base")
        post_base = require_path(args.post_base, "post_base")

        auto_pre, auto_post = auto_find_paths(prefix3, pre_base, post_base)
        pre_path = require_path(auto_pre, "pre_ct(auto)")
        post_path = require_path(auto_post, "post_ct(auto)")

        if args.out_dir is not None:
            out_dir = Path(args.out_dir)
        else:
            out_base = require_path(args.out_base, "out_base")
            out_dir = out_base / "registration_outputs" / prefix3
        out_dir.mkdir(parents=True, exist_ok=True)

    iso = (float(args.iso), float(args.iso), float(args.iso))

    fixed_ct = read_image(pre_path, sitk.sitkFloat32)
    moving_ct = read_image(post_path, sitk.sitkFloat32)

    fixed_iso = resample_isotropic(fixed_ct, out_spacing=iso, is_label=False, default_value=0.0)
    moving_iso = resample_isotropic(moving_ct, out_spacing=iso, is_label=False, default_value=0.0)

    fixed_bone = to_uint8_bone_mask(fixed_iso)
    moving_bone = to_uint8_bone_mask(moving_iso)
    sitk.WriteImage(fixed_bone, str(out_dir / "fixed_bone_mask.nii.gz"))
    sitk.WriteImage(moving_bone, str(out_dir / "moving_bone_mask.nii.gz"))

    final_tfm = register_rigid_affine_mi(
        fixed_ct=fixed_iso,
        moving_ct=moving_iso,
        fixed_bone=fixed_bone,
        moving_bone=moving_bone,
        use_moving_mask=bool(args.use_moving_mask),
        enable_mask_init=(not args.disable_mask_init),
        out_dir=out_dir
    )
    sitk.WriteTransform(final_tfm, str(out_dir / "final_transform.tfm"))

    reg_post = sitk.Resample(moving_iso, fixed_iso, final_tfm, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    reg_post = sitk.Clamp(reg_post, sitk.sitkFloat32, 0.0, 1.0)
    sitk.WriteImage(reg_post, str(out_dir / "registered_postop_ct.nii.gz"))

    fixed_arr = sitk.GetArrayFromImage(fixed_iso)
    moving_arr = sitk.GetArrayFromImage(moving_iso)
    reg_arr = sitk.GetArrayFromImage(reg_post)
    fixed_bone_arr = (sitk.GetArrayFromImage(fixed_bone) > 0)

    ncc = image_ncc(fixed_arr, reg_arr, fixed_bone_arr)

    qc_path = save_qc_png(out_dir, fixed_arr, moving_arr, reg_arr, fixed_bone_arr.astype(np.uint8),
                          save_png=bool(args.qc_png), show=bool(args.display_images))

    affine_det = None
    try:
        t_aff = sitk.AffineTransform(final_tfm)
        m = np.array(t_aff.GetMatrix(), dtype=float).reshape(3, 3)
        affine_det = float(np.linalg.det(m))
    except Exception:
        pass

    report = {
        "pre_ct": str(pre_path),
        "post_ct": str(post_path),
        "out_dir": str(out_dir),
        "isotropic_spacing_mm": float(args.iso),
        "bone_thr_norm": float(BONE_THR),
        "bone_hi_norm_hardcoded": float(BONE_HI),
        "mask_shape_init_enabled": bool(not args.disable_mask_init),
        "use_moving_mask_for_metric": bool(args.use_moving_mask),
        "ncc_on_fixed_bone_roi": ncc,
        "ncc_threshold": float(args.qc_ncc_threshold),
        "qc_flag_manual_review": bool(np.isfinite(ncc) and (ncc < args.qc_ncc_threshold)),
        "final_affine_det_approx": affine_det,
        "qc_preview_png": str(qc_path) if qc_path is not None else None,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] Saved outputs to: {out_dir}")
    print(f"     Bone mask: thr={BONE_THR:.2f}, hi(HARD)={BONE_HI:.2f}")
    print(f"     NCC(on fixed bone ROI) = {ncc:.4f}")
    if affine_det is not None:
        print(f"     Affine det ≈ {affine_det:.4f} (det!=1 implies scaling)")
    if report["qc_flag_manual_review"]:
        print(f"     [WARN] NCC < {args.qc_ncc_threshold:.2f} => manual review suggested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
