import os
import re
import json
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import SimpleITK as sitk

try:
    import napari
    from magicgui import magicgui
except Exception as e:
    print(f"Missing GUI deps for Napari QC: {e}")
    napari = None
    magicgui = None


# ====== Bone params (ONLY for picking an informative QC slice) ======
BONE_THR = 0.19
BONE_HI  = 0.90


# --------------------------
# Basic helpers
# --------------------------
def to_LPS(img: sitk.Image) -> sitk.Image:
    return sitk.DICOMOrient(img, "LPS")


def read_sitk(path: Path, pixel_type=sitk.sitkFloat32) -> Optional[sitk.Image]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        img = sitk.ReadImage(str(path))
        return sitk.Cast(img, pixel_type)
    except Exception:
        return None


def spacing_zyx(img: sitk.Image) -> Tuple[float, float, float]:
    sx, sy, sz = img.GetSpacing()  # x,y,z
    return (sz, sy, sx)            # z,y,x for napari


def auto_contrast(arr: np.ndarray) -> Tuple[float, float]:
    v1, v99 = np.percentile(arr.astype(np.float32), [1, 99])
    if not np.isfinite(v1) or not np.isfinite(v99) or v99 <= v1:
        v1, v99 = float(arr.min()), float(arr.max())
    return float(v1), float(v99)


def extract_prefix3_from_filename(name: str) -> Optional[str]:
    base = os.path.basename(name)
    if len(base) < 3:
        return None
    return base[:3]


def list_prefixes_from_pre_dir(pre_dir: Path) -> List[str]:
    files = [f for f in os.listdir(pre_dir) if f.lower().endswith((".nii", ".nii.gz", ".mha"))]
    prefixes = sorted({extract_prefix3_from_filename(f) for f in files if len(f) >= 3})
    return [p for p in prefixes if p is not None]


def find_pre_ct_by_prefix(pre_dir: Path, prefix3: str) -> Optional[Path]:
    cands: List[Path] = []
    for f in os.listdir(pre_dir):
        if f.startswith("."):
            continue
        if not f.lower().endswith((".nii", ".nii.gz", ".mha")):
            continue
        if f[:3] != prefix3:
            continue
        cands.append(pre_dir / f)

    if not cands:
        return None

    # deterministic preference: _0000 > not-mask > larger size
    def score(p: Path) -> float:
        s = 0.0
        name = p.name.lower()
        if "_0000" in name:
            s += 100.0
        if any(k in name for k in ["mask", "seg", "label", "resection", "cavity"]):
            s -= 200.0
        try:
            s += min(p.stat().st_size / (1024 * 1024), 500.0) * 0.1
        except Exception:
            pass
        return s

    cands.sort(key=score, reverse=True)
    return cands[0]


# --------------------------
# Deterministic discovery of registration outputs (NO assumptions)
# --------------------------
def _parse_case_prefix_from_report(report: dict) -> Optional[str]:
    for k in ["case_prefix3", "case_id", "case", "subject_id"]:
        v = report.get(k, None)
        if isinstance(v, str) and len(v) >= 3:
            return v[:3]
    pre_ct = report.get("pre_ct", None)
    if isinstance(pre_ct, str):
        bn = os.path.basename(pre_ct)
        if len(bn) >= 3:
            return bn[:3]
    return None


def index_registration_outputs(out_root: Path, strict: bool = True) -> Dict[str, Dict]:
    """
    Valid case_dir MUST contain:
      - registered_postop_ct.nii.gz
      - final_transform.tfm
      - report.json   (required if strict=True)
    Case prefix is determined from report.json (recommended).
    """
    out_root = Path(out_root)
    if not out_root.exists():
        raise FileNotFoundError(f"out_root not found: {out_root}")

    mapping: Dict[str, Dict] = {}
    reports = list(out_root.rglob("report.json"))

    def try_add(case_dir: Path, report_path: Optional[Path]):
        reg_ct = case_dir / "registered_postop_ct.nii.gz"
        tfm = case_dir / "final_transform.tfm"
        if not reg_ct.exists() or not tfm.exists():
            return

        prefix3 = None
        if report_path is not None and report_path.exists():
            try:
                rep = json.loads(report_path.read_text(encoding="utf-8"))
                prefix3 = _parse_case_prefix_from_report(rep)
            except Exception:
                prefix3 = None

        if prefix3 is None:
            if strict:
                return
            # non-strict fallback: first 3 digits in dirname
            m = re.search(r"(\d{3})", case_dir.name)
            if m:
                prefix3 = m.group(1)

        if prefix3 is None:
            return

        cand_mtime = reg_ct.stat().st_mtime
        prev = mapping.get(prefix3, None)
        if prev is None:
            mapping[prefix3] = {"case_dir": case_dir, "reg_ct": reg_ct, "final_tfm": tfm, "report_json": report_path}
        else:
            prev_mtime = Path(prev["reg_ct"]).stat().st_mtime
            if cand_mtime > prev_mtime:
                mapping[prefix3] = {"case_dir": case_dir, "reg_ct": reg_ct, "final_tfm": tfm, "report_json": report_path}

    for rp in reports:
        try_add(rp.parent, rp)

    if not strict:
        for reg_ct in out_root.rglob("registered_postop_ct.nii.gz"):
            case_dir = reg_ct.parent
            rp = case_dir / "report.json"
            try_add(case_dir, rp if rp.exists() else None)

    return mapping


# --------------------------
# Manual affine delta (translation + rotation + scaling)
# --------------------------
def _rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [0,   0,  1]], dtype=np.float64)

    # Apply X then Y then Z: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx


def _fixed_physical_center(img: sitk.Image) -> Tuple[float, float, float]:
    # size in x,y,z
    sx, sy, sz = img.GetSize()
    center_index = [(sx - 1) / 2.0, (sy - 1) / 2.0, (sz - 1) / 2.0]
    center_point = img.TransformContinuousIndexToPhysicalPoint(center_index)
    return (float(center_point[0]), float(center_point[1]), float(center_point[2]))


def build_manual_affine_delta(
    fixed_ref: sitk.Image,
    shift_zyx_mm: Tuple[float, float, float],
    rot_xyz_deg: Tuple[float, float, float],
    scale_xyz: Tuple[float, float, float],
) -> sitk.AffineTransform:
    """
    Napari shifts are provided as (z,y,x) in mm -> convert to xyz.
    We save delta such that baking matches “napari positive shift” effect:
      - translation in sitk is (-shift_xyz)
    Rotation/scaling are applied around the fixed image physical center.

    Model:
      p_out -> p_in =  M * (p_out - c) + c + t
    Where:
      M = R @ S  (scale first, then rotate)
      t = (-shift_x, -shift_y, -shift_z)
    """
    shift_z, shift_y, shift_x = [float(v) for v in shift_zyx_mm]
    sx, sy, sz = [float(v) for v in scale_xyz]
    rx, ry, rz = [float(v) for v in rot_xyz_deg]

    # scale then rotate
    S = np.diag([sx, sy, sz]).astype(np.float64)
    R = _rotation_matrix_xyz(rx, ry, rz)
    M = R @ S  # 3x3

    delta = sitk.AffineTransform(3)
    delta.SetCenter(_fixed_physical_center(fixed_ref))
    delta.SetMatrix(tuple(M.reshape(-1).tolist()))

    # IMPORTANT sign to match napari translate
    delta.SetTranslation((-shift_x, -shift_y, -shift_z))
    return delta


def apply_affine_and_get_image(
    fixed_ref: sitk.Image,
    reg_img_in_fixed: sitk.Image,
    delta_affine: sitk.Transform,
    clamp01: bool = True,
) -> sitk.Image:
    out = sitk.Resample(reg_img_in_fixed, fixed_ref, delta_affine, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    if clamp01:
        out = sitk.Clamp(out, sitk.sitkFloat32, 0.0, 1.0)
    return out


def save_manual_outputs(
    case_dir: Path,
    prefix3: str,
    fixed_ref: sitk.Image,
    reg_img_in_fixed: sitk.Image,
    shift_zyx_mm: Tuple[float, float, float],
    rot_xyz_deg: Tuple[float, float, float],
    scale_xyz: Tuple[float, float, float],
    overwrite: bool = False,
    save_png: bool = True,
) -> Dict:
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    delta = build_manual_affine_delta(fixed_ref, shift_zyx_mm, rot_xyz_deg, scale_xyz)
    reg_manual = apply_affine_and_get_image(fixed_ref, reg_img_in_fixed, delta, clamp01=True)

    out_default = case_dir / "registered_postop_ct.nii.gz"
    out_manual  = case_dir / "registered_postop_ct_manual.nii.gz"
    out_tfm     = case_dir / "manual_delta_affine.tfm"
    out_json    = case_dir / "manual_qc.json"
    out_png     = case_dir / "qc_manual.png"

    out_img = out_default if overwrite else out_manual
    sitk.WriteImage(reg_manual, str(out_img))
    sitk.WriteTransform(delta, str(out_tfm))

    qc_png_path = None
    if save_png:
        try:
            import matplotlib.pyplot as plt
            fixed_arr = sitk.GetArrayFromImage(fixed_ref)
            reg_arr   = sitk.GetArrayFromImage(reg_img_in_fixed)
            man_arr   = sitk.GetArrayFromImage(reg_manual)

            bone = np.logical_and(fixed_arr > BONE_THR, fixed_arr <= BONE_HI)
            z = int(np.argmax(bone.reshape(bone.shape[0], -1).sum(axis=1))) if bone.any() else fixed_arr.shape[0] // 2

            fig = plt.figure(figsize=(14, 4))
            ax1 = fig.add_subplot(1, 3, 1); ax1.imshow(fixed_arr[z], cmap="gray"); ax1.set_title("Fixed"); ax1.axis("off")
            ax2 = fig.add_subplot(1, 3, 2); ax2.imshow(reg_arr[z], cmap="gray"); ax2.set_title("Registered"); ax2.axis("off")
            ax3 = fig.add_subplot(1, 3, 3); ax3.imshow(man_arr[z], cmap="gray"); ax3.set_title("Manual"); ax3.axis("off")
            fig.savefig(str(out_png), dpi=140, bbox_inches="tight")
            plt.close(fig)
            qc_png_path = out_png
        except Exception:
            qc_png_path = None

    sz, sy, sx = [float(v) for v in shift_zyx_mm]
    rx, ry, rz = [float(v) for v in rot_xyz_deg]
    scx, scy, scz = [float(v) for v in scale_xyz]

    meta = {
        "case_prefix3": prefix3,
        "manual_params": {
            "shift_mm_zyx": [sz, sy, sx],
            "rotation_deg_xyz": [rx, ry, rz],
            "scale_xyz": [scx, scy, scz],
            "center_xyz_mm": list(_fixed_physical_center(fixed_ref)),
            "matrix_model": "p_in = (R@S)*(p_out-center) + center + t, t = (-shift_xyz)",
        },
        "outputs": {
            "output_image": str(out_img),
            "delta_transform_tfm": str(out_tfm),
            "qc_png": str(qc_png_path) if qc_png_path else None,
        }
    }
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# --------------------------
# Napari QC main
# --------------------------
def run_qc_viewer(pre_dir: Path, out_root: Path, strict: bool = True):
    if napari is None or magicgui is None:
        raise SystemExit("Install: pip install napari[all] magicgui pyqt5 matplotlib SimpleITK numpy")

    pre_dir = Path(pre_dir)
    out_root = Path(out_root)

    prefixes_pre = list_prefixes_from_pre_dir(pre_dir)
    out_index = index_registration_outputs(out_root, strict=strict)

    prefixes = sorted(set(prefixes_pre).intersection(out_index.keys()))
    if not prefixes:
        raise RuntimeError(
            "No QC-able cases.\n"
            f"- prefixes in pre_dir: {len(prefixes_pre)}\n"
            f"- valid cases in out_root (strict={strict}): {len(out_index)}\n"
            "Make sure each case_dir contains: registered_postop_ct.nii.gz + final_transform.tfm + report.json."
        )

    viewer = napari.Viewer(title="Manual QC for registered_postop_ct (Affine delta export)")

    # state
    state: Dict[str, Optional[object]] = {"fixed": None, "reg": None, "case_dir": None, "prefix": None}

    def load_case(prefix3: str):
        rec = out_index[prefix3]
        case_dir = Path(rec["case_dir"])
        reg_ct = Path(rec["reg_ct"])

        pre_ct = find_pre_ct_by_prefix(pre_dir, prefix3)
        if pre_ct is None:
            raise RuntimeError(f"Missing pre CT for prefix={prefix3}")

        fixed = read_sitk(pre_ct, sitk.sitkFloat32)
        reg = read_sitk(reg_ct, sitk.sitkFloat32)
        if fixed is None or reg is None:
            raise RuntimeError("Failed to read images.")

        fixed = to_LPS(fixed)
        reg = to_LPS(reg)

        # Ensure on same grid
        reg = sitk.Resample(reg, fixed, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)

        state["fixed"] = fixed
        state["reg"] = reg
        state["case_dir"] = case_dir
        state["prefix"] = prefix3
        return fixed, reg, pre_ct, reg_ct, case_dir

    fixed0, reg0, _, _, _ = load_case(prefixes[0])
    arr_f = sitk.GetArrayFromImage(fixed0)
    arr_r = sitk.GetArrayFromImage(reg0)
    scale = spacing_zyx(fixed0)

    cl_f = auto_contrast(arr_f)
    cl_r = auto_contrast(arr_r)
    cl = (min(cl_f[0], cl_r[0]), max(cl_f[1], cl_r[1]))

    l_fixed = viewer.add_image(arr_f, name=f"Fixed(pre): {prefixes[0]}", scale=scale, colormap="gray", contrast_limits=cl)
    try:
        l_fixed.locked = True
    except Exception:
        pass

    l_red = viewer.add_image(
        arr_r,
        name="Registered post (red)",
        scale=scale,
        colormap="red",
        blending="additive",
        opacity=0.5,
        contrast_limits=cl
    )

    viewer.dims.order = (1, 0, 2)

    # --- UI ---
    @magicgui(
        layout="vertical",
        case_prefix={"choices": prefixes, "label": "Case (first 3 chars)"},
        red_opacity={"min": 0, "max": 1, "step": 0.05, "label": "Red opacity", "value": 0.5},
        auto_window={"label": "Auto window (1–99%)", "value": True},

        sep0={"widget_type": "Label", "label": "--- Fast translate (mm) for quick viewing ---"},
        shift_z={"min": -50, "max": 50, "step": 0.5, "label": "Shift Z (mm)", "value": 0.0},
        shift_y={"min": -50, "max": 50, "step": 0.5, "label": "Shift Y (mm)", "value": 0.0},
        shift_x={"min": -50, "max": 50, "step": 0.5, "label": "Shift X (mm)", "value": 0.0},

        sep1={"widget_type": "Label", "label": "--- Affine delta (applied on Preview/Save) ---"},
        rot_x={"min": -15, "max": 15, "step": 0.25, "label": "Rotate X (deg)", "value": 0.0},
        rot_y={"min": -15, "max": 15, "step": 0.25, "label": "Rotate Y (deg)", "value": 0.0},
        rot_z={"min": -15, "max": 15, "step": 0.25, "label": "Rotate Z (deg)", "value": 0.0},
        scale_x={"min": 0.90, "max": 1.10, "step": 0.005, "label": "Scale X", "value": 1.0},
        scale_y={"min": 0.90, "max": 1.10, "step": 0.005, "label": "Scale Y", "value": 1.0},
        scale_z={"min": 0.90, "max": 1.10, "step": 0.005, "label": "Scale Z", "value": 1.0},

        preview_btn={"widget_type": "PushButton", "text": "Preview (resample once)"},
        save_btn={"widget_type": "PushButton", "text": "Save manual outputs"},
        overwrite={"label": "Overwrite registered_postop_ct.nii.gz", "value": False},
        save_png={"label": "Also save qc_manual.png", "value": True},
        reset_btn={"widget_type": "PushButton", "text": "Reset params"},
    )
    def widget(
        case_prefix: str,
        red_opacity=0.5,
        auto_window=True,

        sep0="---",
        shift_z=0.0, shift_y=0.0, shift_x=0.0,

        sep1="---",
        rot_x=0.0, rot_y=0.0, rot_z=0.0,
        scale_x=1.0, scale_y=1.0, scale_z=1.0,

        preview_btn=False,
        save_btn=False,
        overwrite=False,
        save_png=True,
        reset_btn=False,
    ):
        # Switch case
        target = f"Fixed(pre): {case_prefix}"
        if l_fixed.name != target:
            fixed, reg, _, _, _ = load_case(case_prefix)
            arr_f = sitk.GetArrayFromImage(fixed)
            arr_r = sitk.GetArrayFromImage(reg)

            l_fixed.data = arr_f
            l_fixed.scale = spacing_zyx(fixed)
            l_fixed.name = target

            l_red.data = arr_r
            l_red.scale = spacing_zyx(fixed)

            # reset params on case switch
            widget.shift_z.value = 0.0
            widget.shift_y.value = 0.0
            widget.shift_x.value = 0.0
            widget.rot_x.value = 0.0
            widget.rot_y.value = 0.0
            widget.rot_z.value = 0.0
            widget.scale_x.value = 1.0
            widget.scale_y.value = 1.0
            widget.scale_z.value = 1.0

            if auto_window:
                cl_f = auto_contrast(arr_f)
                cl_r = auto_contrast(arr_r)
                cl2 = (min(cl_f[0], cl_r[0]), max(cl_f[1], cl_r[1]))
                l_fixed.contrast_limits = cl2
                l_red.contrast_limits = cl2

        # Live: opacity + fast translate only
        l_red.opacity = red_opacity
        l_red.translate = (float(shift_z), float(shift_y), float(shift_x))

        if auto_window:
            arr_f = l_fixed.data
            arr_r = l_red.data
            cl_f = auto_contrast(arr_f)
            cl_r = auto_contrast(arr_r)
            cl2 = (min(cl_f[0], cl_r[0]), max(cl_f[1], cl_r[1]))
            l_fixed.contrast_limits = cl2
            l_red.contrast_limits = cl2

    @widget.reset_btn.changed.connect
    def _reset():
        widget.shift_z.value = 0.0
        widget.shift_y.value = 0.0
        widget.shift_x.value = 0.0
        widget.rot_x.value = 0.0
        widget.rot_y.value = 0.0
        widget.rot_z.value = 0.0
        widget.scale_x.value = 1.0
        widget.scale_y.value = 1.0
        widget.scale_z.value = 1.0

    def _bake_in_memory():
        fixed = state["fixed"]
        reg = state["reg"]
        if fixed is None or reg is None:
            print("[PREVIEW] No case loaded.")
            return None

        delta = build_manual_affine_delta(
            fixed_ref=fixed,
            shift_zyx_mm=(float(widget.shift_z.value), float(widget.shift_y.value), float(widget.shift_x.value)),
            rot_xyz_deg=(float(widget.rot_x.value), float(widget.rot_y.value), float(widget.rot_z.value)),
            scale_xyz=(float(widget.scale_x.value), float(widget.scale_y.value), float(widget.scale_z.value)),
        )
        out = apply_affine_and_get_image(fixed, reg, delta, clamp01=True)
        return out

    @widget.preview_btn.changed.connect
    def _preview():
        out = _bake_in_memory()
        if out is None:
            return
        # Update red layer with baked preview, and reset fast translate to 0
        state["reg"] = out
        l_red.data = sitk.GetArrayFromImage(out)

        widget.shift_z.value = 0.0
        widget.shift_y.value = 0.0
        widget.shift_x.value = 0.0
        # keep rot/scale as-is so user can still see params; optional: reset if you prefer

        print("[PREVIEW] Applied affine delta in memory (one resample).")

    @widget.save_btn.changed.connect
    def _save():
        fixed = state["fixed"]
        reg = state["reg"]
        case_dir = state["case_dir"]
        prefix3 = state["prefix"]
        if fixed is None or reg is None or case_dir is None or prefix3 is None:
            print("[SAVE] No case loaded.")
            return

        meta = save_manual_outputs(
            case_dir=Path(case_dir),
            prefix3=str(prefix3),
            fixed_ref=fixed,
            reg_img_in_fixed=reg,
            shift_zyx_mm=(float(widget.shift_z.value), float(widget.shift_y.value), float(widget.shift_x.value)),
            rot_xyz_deg=(float(widget.rot_x.value), float(widget.rot_y.value), float(widget.rot_z.value)),
            scale_xyz=(float(widget.scale_x.value), float(widget.scale_y.value), float(widget.scale_z.value)),
            overwrite=bool(widget.overwrite.value),
            save_png=bool(widget.save_png.value),
        )
        print(f"[SAVE] OK: {meta['outputs']['output_image']}")
        print(f"       tfm: {meta['outputs']['delta_transform_tfm']}")

        # After save, load saved image into red layer and reset translate (so view matches file)
        saved_path = Path(meta["outputs"]["output_image"])
        new_reg = read_sitk(saved_path, sitk.sitkFloat32)
        if new_reg is not None:
            new_reg = to_LPS(new_reg)
            new_reg = sitk.Resample(new_reg, fixed, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
            state["reg"] = new_reg
            l_red.data = sitk.GetArrayFromImage(new_reg)

        # reset fast translate
        widget.shift_z.value = 0.0
        widget.shift_y.value = 0.0
        widget.shift_x.value = 0.0

    viewer.window.add_dock_widget(widget, area="right")
    napari.run()


def parse_args():
    ap = argparse.ArgumentParser(description="Napari manual QC for registered_postop_ct with affine delta export.")
    ap.add_argument("--pre_dir", type=Path, required=True, help="Directory containing pre-op CTs (used as fixed)")
    ap.add_argument("--out_root", type=Path, required=True, help="Root directory containing auto registration outputs")
    ap.add_argument("--strict", action="store_true", default=False,
                    help="Strict mode: require report.json + final_transform.tfm + registered_postop_ct.nii.gz")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_qc_viewer(args.pre_dir, args.out_root, strict=bool(args.strict))
