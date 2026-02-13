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
    print(f"缺少依赖(用于GUI): {e}")
    napari = None
    magicgui = None


# --------------------------
# For QC slice pick only (optional)
# --------------------------
BONE_THR = 0.19
BONE_HI  = 0.90


# --------------------------
# Basic helpers
# --------------------------
def to_LPS(img: sitk.Image) -> sitk.Image:
    return sitk.DICOMOrient(img, "LPS")


def read_sitk(path: Path, pixel_type=sitk.sitkFloat32) -> Optional[sitk.Image]:
    if path is None or (not Path(path).exists()):
        return None
    try:
        img = sitk.ReadImage(str(path))
        return sitk.Cast(img, pixel_type)
    except Exception:
        return None


def spacing_zyx(img: sitk.Image) -> Tuple[float, float, float]:
    sx, sy, sz = img.GetSpacing()
    return (sz, sy, sx)


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
    # Prefer explicit fields if present
    for k in ["case_prefix3", "case_id", "case", "subject_id"]:
        v = report.get(k, None)
        if isinstance(v, str) and len(v) >= 3:
            return v[:3]

    # fallback: if pre_ct path exists in report, use its filename prefix
    pre_ct = report.get("pre_ct", None)
    if isinstance(pre_ct, str) and len(os.path.basename(pre_ct)) >= 3:
        return os.path.basename(pre_ct)[:3]

    return None


def _prefix_from_dirname(p: Path) -> Optional[str]:
    # deterministic: if dirname starts with 3 digits -> use them
    m = re.match(r"^(\d{3})", p.name)
    if m:
        return m.group(1)
    # else, if has any 3-digit substring, take first
    m2 = re.search(r"(\d{3})", p.name)
    if m2:
        return m2.group(1)
    return None


def index_registration_outputs(out_root: Path, strict: bool = True) -> Dict[str, Dict]:
    """
    Build mapping: prefix3 -> {case_dir, reg_ct, final_tfm, report_json}
    Deterministic rule for a valid case_dir:
      - must contain registered_postop_ct.nii.gz
      - must contain final_transform.tfm
      - must contain report.json (if strict=True)
    Case id is determined by:
      - report.json field (case_prefix3/case_id/pre_ct filename prefix), else
      - case_dir name (only used when strict=False)
    """
    out_root = Path(out_root)
    if not out_root.exists():
        raise FileNotFoundError(f"out_root not found: {out_root}")

    # find all report.json first (preferred)
    reports = list(out_root.rglob("report.json"))
    mapping: Dict[str, Dict] = {}

    def try_add(case_dir: Path, report_path: Optional[Path]):
        reg_ct = case_dir / "registered_postop_ct.nii.gz"
        tfm = case_dir / "final_transform.tfm"
        if not reg_ct.exists() or not tfm.exists():
            return

        prefix3 = None
        report_obj = None
        if report_path is not None and report_path.exists():
            try:
                report_obj = json.loads(report_path.read_text(encoding="utf-8"))
                prefix3 = _parse_case_prefix_from_report(report_obj)
            except Exception:
                prefix3 = None

        if prefix3 is None:
            if strict:
                return  # strict mode: must be determinable from report
            prefix3 = _prefix_from_dirname(case_dir)

        if prefix3 is None:
            return

        # handle duplicates deterministically: pick newest report/ct by mtime
        prev = mapping.get(prefix3)
        cand_mtime = reg_ct.stat().st_mtime
        if prev is None:
            mapping[prefix3] = {
                "case_dir": case_dir,
                "reg_ct": reg_ct,
                "final_tfm": tfm,
                "report_json": report_path,
            }
        else:
            prev_mtime = Path(prev["reg_ct"]).stat().st_mtime
            if cand_mtime > prev_mtime:
                mapping[prefix3] = {
                    "case_dir": case_dir,
                    "reg_ct": reg_ct,
                    "final_tfm": tfm,
                    "report_json": report_path,
                }

    # index from reports
    for rp in reports:
        case_dir = rp.parent
        try_add(case_dir, rp)

    if not strict:
        # also consider case dirs that only have files but no report.json
        reg_files = list(out_root.rglob("registered_postop_ct.nii.gz"))
        for reg_ct in reg_files:
            case_dir = reg_ct.parent
            rp = case_dir / "report.json"
            try_add(case_dir, rp if rp.exists() else None)

    return mapping


# --------------------------
# Manual output
# --------------------------
def apply_manual_translation_and_save(
    fixed_ref: sitk.Image,
    reg_img: sitk.Image,
    case_dir: Path,
    prefix3: str,
    shift_zyx_mm: Tuple[float, float, float],
    overwrite: bool = False,
    save_png: bool = True
) -> Dict:
    """
    Napari layer.translate uses (z,y,x) in world units (mm if scale set).
    Positive translate shifts the layer forward; to bake into voxels we resample with Translation(-shift_xyz).
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    shift_z, shift_y, shift_x = [float(v) for v in shift_zyx_mm]
    shift_xyz = (shift_x, shift_y, shift_z)

    delta = sitk.TranslationTransform(3, (-shift_xyz[0], -shift_xyz[1], -shift_xyz[2]))

    reg_manual = sitk.Resample(
        reg_img, fixed_ref, delta,
        sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )

    out_default = case_dir / "registered_postop_ct.nii.gz"
    out_manual  = case_dir / "registered_postop_ct_manual.nii.gz"
    out_tfm     = case_dir / "manual_delta_translation.tfm"
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
            reg_arr = sitk.GetArrayFromImage(reg_img)
            man_arr = sitk.GetArrayFromImage(reg_manual)
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

    meta = {
        "case_prefix3": prefix3,
        "shift_mm_zyx": [shift_z, shift_y, shift_x],
        "shift_mm_xyz": [shift_xyz[0], shift_xyz[1], shift_xyz[2]],
        "delta_transform_note": "Saved delta is Translation(-shift_xyz) to reproduce napari translate effect.",
        "output_image": str(out_img),
        "delta_transform_tfm": str(out_tfm),
        "qc_png": str(qc_png_path) if qc_png_path else None,
    }
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# --------------------------
# Napari main
# --------------------------
def run_qc_viewer(pre_dir: Path, out_root: Path, strict: bool = True):
    if napari is None or magicgui is None:
        raise SystemExit("请先安装: pip install napari[all] magicgui pyqt5 matplotlib")

    pre_dir = Path(pre_dir)
    out_root = Path(out_root)

    prefixes_pre = list_prefixes_from_pre_dir(pre_dir)
    out_index = index_registration_outputs(out_root, strict=strict)

    prefixes = sorted(set(prefixes_pre).intersection(out_index.keys()))
    if not prefixes:
        raise RuntimeError(
            "没有可核查病例。\n"
            f"- 术前目录prefix数: {len(prefixes_pre)}\n"
            f"- 配准输出可识别case数: {len(out_index)} (strict={strict})\n"
            "请确认 out_root 下存在每例的 registered_postop_ct.nii.gz + final_transform.tfm + report.json。"
        )

    viewer = napari.Viewer(title="自动配准结果人工核查（只读加载 + 手动输出）")

    state = {"fixed": None, "reg": None, "case_dir": None}

    def load_case(prefix3: str):
        rec = out_index[prefix3]
        case_dir = Path(rec["case_dir"])
        reg_ct = Path(rec["reg_ct"])

        pre_ct = find_pre_ct_by_prefix(pre_dir, prefix3)
        if pre_ct is None:
            raise RuntimeError(f"术前找不到 prefix={prefix3} 的CT")

        fixed = read_sitk(pre_ct, sitk.sitkFloat32)
        reg = read_sitk(reg_ct, sitk.sitkFloat32)
        if fixed is None or reg is None:
            raise RuntimeError("读取影像失败")

        fixed = to_LPS(fixed)
        reg = to_LPS(reg)

        # safety resample to fixed grid
        reg = sitk.Resample(reg, fixed, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)

        state["fixed"] = fixed
        state["reg"] = reg
        state["case_dir"] = case_dir
        return fixed, reg, pre_ct, reg_ct

    fixed0, reg0, pre0, regp0 = load_case(prefixes[0])
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

    l_red = viewer.add_image(arr_r, name="Registered post (red)", scale=scale, colormap="red",
                             blending="additive", opacity=0.5, contrast_limits=cl)

    viewer.dims.order = (1, 0, 2)

    @magicgui(
        layout="vertical",
        case_prefix={"choices": prefixes, "label": "病例ID (前3位)"},
        red_opacity={"min": 0, "max": 1, "step": 0.05, "label": "红CT 透明度", "value": 0.5},
        auto_window={"label": "自动窗宽(1-99%)", "value": True},
        sep_line={"widget_type": "Label", "label": "--- 微调 (mm) + 输出 ---"},
        shift_z={"min": -50, "max": 50, "step": 0.5, "label": "Z 平移(mm)", "value": 0},
        shift_y={"min": -50, "max": 50, "step": 0.5, "label": "Y 平移(mm)", "value": 0},
        shift_x={"min": -50, "max": 50, "step": 0.5, "label": "X 平移(mm)", "value": 0},
        overwrite={"label": "覆盖 registered_postop_ct.nii.gz", "value": False},
        save_png={"label": "输出 qc_manual.png", "value": True},
        save_btn={"widget_type": "PushButton", "text": "保存当前微调结果"},
        reset_btn={"widget_type": "PushButton", "text": "重置微调"},
    )
    def widget(case_prefix: str, red_opacity=0.5, auto_window=True, sep_line="---",
               shift_z=0.0, shift_y=0.0, shift_x=0.0,
               overwrite=False, save_png=True, save_btn=False, reset_btn=False):

        target = f"Fixed(pre): {case_prefix}"
        if l_fixed.name != target:
            fixed, reg, _, _ = load_case(case_prefix)
            arr_f = sitk.GetArrayFromImage(fixed)
            arr_r = sitk.GetArrayFromImage(reg)

            l_fixed.data = arr_f
            l_fixed.scale = spacing_zyx(fixed)
            l_fixed.name = target

            l_red.data = arr_r
            l_red.scale = spacing_zyx(fixed)

            widget.shift_z.value = 0
            widget.shift_y.value = 0
            widget.shift_x.value = 0

            if auto_window:
                cl_f = auto_contrast(arr_f)
                cl_r = auto_contrast(arr_r)
                cl2 = (min(cl_f[0], cl_r[0]), max(cl_f[1], cl_r[1]))
                l_fixed.contrast_limits = cl2
                l_red.contrast_limits = cl2

        l_red.opacity = red_opacity
        l_red.translate = (shift_z, shift_y, shift_x)

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
        widget.shift_z.value = 0
        widget.shift_y.value = 0
        widget.shift_x.value = 0

    @widget.save_btn.changed.connect
    def _save():
        prefix3 = widget.case_prefix.value
        fixed = state["fixed"]
        reg = state["reg"]
        case_dir = state["case_dir"]
        if fixed is None or reg is None or case_dir is None:
            print("[SAVE] 当前病例未加载，无法保存。")
            return

        meta = apply_manual_translation_and_save(
            fixed_ref=fixed,
            reg_img=reg,
            case_dir=Path(case_dir),
            prefix3=prefix3,
            shift_zyx_mm=(float(widget.shift_z.value), float(widget.shift_y.value), float(widget.shift_x.value)),
            overwrite=bool(widget.overwrite.value),
            save_png=bool(widget.save_png.value)
        )
        print(f"[SAVE] OK: {meta['output_image']}")
        print(f"       delta: {meta['delta_transform_tfm']}")

        # reload saved image into red layer, reset translate to 0 so view matches file
        new_reg = read_sitk(Path(meta["output_image"]), sitk.sitkFloat32)
        if new_reg is not None:
            new_reg = to_LPS(new_reg)
            new_reg = sitk.Resample(new_reg, fixed, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
            state["reg"] = new_reg
            l_red.data = sitk.GetArrayFromImage(new_reg)

        widget.shift_z.value = 0
        widget.shift_y.value = 0
        widget.shift_x.value = 0

    viewer.window.add_dock_widget(widget, area="right")
    napari.run()


def parse_args():
    ap = argparse.ArgumentParser(description="Napari QC viewer for registered_postop_ct outputs (deterministic discovery).")
    ap.add_argument("--pre_dir", type=Path, required=True, help="术前CT目录（用于找prefix列表和加载fixed）")
    ap.add_argument("--out_root", type=Path, required=True, help="配准输出根目录（递归扫描report.json确定case）")
    ap.add_argument("--strict", action="store_true", default=False,
                    help="严格模式：只接受同时包含 report.json + final_transform.tfm + registered_postop_ct.nii.gz 的case")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # 默认 strict=False 是为了更兼容旧输出；但你强调可复现，建议你运行时加 --strict
    run_qc_viewer(args.pre_dir, args.out_root, strict=bool(args.strict))
