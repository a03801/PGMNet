#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Napari viewer: overlay CT + (AI mask) + (QC/resection mask) and display coverage ratio.

Coverage ratio definition:
    coverage = |AI ∩ QC| / |AI|

Notes for public release:
- No hard-coded private paths.
- Case matching uses a filename-based case key (default: strip extensions and trailing _0000-like).
- Masks are resampled to the CT geometry (nearest neighbor).
"""

import os
import re
import argparse
from typing import Tuple, Optional, Dict

import numpy as np
import SimpleITK as sitk

# GUI deps (optional)
try:
    import napari
    from magicgui import magicgui
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.patches as patches
    from PyQt5.QtWidgets import QWidget, QVBoxLayout
except Exception as e:
    print(f"[WARN] GUI dependencies missing: {e}")
    napari = None
    magicgui = None


# -----------------------
# Core metric
# -----------------------
def coverage_ratio(ai_mask: sitk.Image, qc_mask: sitk.Image) -> float:
    """
    coverage = (AI ∩ QC) / AI
    Both images must be in the same geometry.
    """
    arr_ai = sitk.GetArrayFromImage(ai_mask) > 0
    arr_qc = sitk.GetArrayFromImage(qc_mask) > 0
    vol_ai = int(np.count_nonzero(arr_ai))
    if vol_ai == 0:
        return 0.0
    inter = np.logical_and(arr_ai, arr_qc)
    vol_inter = int(np.count_nonzero(inter))
    return float(vol_inter / vol_ai)


# -----------------------
# Chart widget
# -----------------------
class CoverageChart(QWidget):
    def __init__(self, threshold: float = 0.72):
        super().__init__()
        self.threshold = float(threshold)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.fig = Figure(figsize=(3, 3), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)
        self.update_chart(0.0)

    def update_chart(self, coverage: float):
        self.ax.clear()
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.35, 1.2)
        self.ax.axis("off")

        # base = AI total
        circle_base = patches.Circle((0, 0), 1.0, facecolor="red", edgecolor="black", linewidth=1.5)
        self.ax.add_patch(circle_base)

        # cover area proportional to coverage -> r = sqrt(coverage)
        radius_cover = float(np.sqrt(max(coverage, 0.0))) if coverage > 0 else 0.0
        circle_cover = patches.Circle((0, 0), radius_cover, facecolor="yellow", edgecolor="black", linewidth=1.0)
        self.ax.add_patch(circle_cover)

        pct = coverage * 100.0
        is_safe = coverage >= self.threshold
        risk_text = f"Low Risk (≥{self.threshold*100:.0f}%)" if is_safe else f"High Risk (<{self.threshold*100:.0f}%)"
        risk_color = "green" if is_safe else "red"

        self.ax.text(0, 0, f"{pct:.1f}%", fontsize=18, fontweight="bold",
                     ha="center", va="center", color="black")
        self.ax.text(0, -1.25, f"Coverage Rate\n{risk_text}", fontsize=10, fontweight="bold",
                     ha="center", va="top", color=risk_color)

        self.canvas.draw()


# -----------------------
# IO helpers
# -----------------------
def sitk_read(path: str) -> sitk.Image:
    return sitk.ReadImage(path)

def to_LPS(img: sitk.Image) -> sitk.Image:
    return sitk.DICOMOrient(img, "LPS")

def resample_to(ref: sitk.Image, moving: sitk.Image, is_label: bool) -> sitk.Image:
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    if is_label:
        moving = sitk.Cast(moving, sitk.sitkUInt8)
    return sitk.Resample(moving, ref, sitk.Transform(), interp, 0, moving.GetPixelID())

def spacing_zyx(img: sitk.Image) -> Tuple[float, float, float]:
    sx, sy, sz = img.GetSpacing()
    return (sz, sy, sx)

def make_case_key(filename: str) -> str:
    """
    Build a deterministic case key from a CT filename.
    - Remove extensions
    - Remove trailing '_dddd' (e.g., '_0000')
    """
    name = os.path.basename(filename)
    name = re.sub(r"\.nii(\.gz)?$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\.mha$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_\d{4}$", "", name)  # remove _0000-like suffix
    return name

def find_first_mask(mask_dir: str, case_key: str) -> Optional[str]:
    """
    Deterministic mask matching:
    - First prefer exact prefix match: filename startswith(case_key)
    - Then fallback: substring match
    """
    if not mask_dir or (not os.path.isdir(mask_dir)):
        return None

    files = [f for f in os.listdir(mask_dir) if f.lower().endswith((".nii", ".nii.gz", ".mha"))]
    if not files:
        return None

    # 1) strict prefix match
    prefix_hits = [f for f in files if f.lower().startswith(case_key.lower())]
    if prefix_hits:
        prefix_hits.sort(key=lambda x: (len(x), x.lower()))
        return os.path.join(mask_dir, prefix_hits[0])

    # 2) fallback substring match
    sub_hits = [f for f in files if case_key.lower() in f.lower()]
    if sub_hits:
        sub_hits.sort(key=lambda x: (len(x), x.lower()))
        return os.path.join(mask_dir, sub_hits[0])

    return None


# -----------------------
# Main viewer
# -----------------------
def run_viewer(img_dir: str, ai_mask_dir: str, qc_mask_dir: str,
               risk_threshold: float = 0.72,
               normalized_01: bool = True):
    if napari is None or magicgui is None:
        raise SystemExit("Please install GUI deps: pip install napari[all] magicgui pyqt5 matplotlib")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"img_dir not found: {img_dir}")

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".nii", ".nii.gz", ".mha"))]
    if not img_files:
        raise SystemExit("No CT images found in img_dir.")

    # map: case_key -> full path
    case_map: Dict[str, str] = {make_case_key(f): os.path.join(img_dir, f) for f in img_files}
    all_cases = sorted(case_map.keys())

    viewer = napari.Viewer(title="Coverage QC Viewer (AI vs QC masks)")
    chart = CoverageChart(threshold=risk_threshold)
    viewer.window.add_dock_widget(chart, area="right", name="Coverage")

    def load_case(case_key: str):
        ct = to_LPS(sitk_read(case_map[case_key]))
        ai_path = find_first_mask(ai_mask_dir, case_key)
        qc_path = find_first_mask(qc_mask_dir, case_key)

        ai = to_LPS(sitk_read(ai_path)) if ai_path else None
        qc = to_LPS(sitk_read(qc_path)) if qc_path else None

        if ai is not None:
            ai = resample_to(ct, ai, is_label=True)
        if qc is not None:
            qc = resample_to(ct, qc, is_label=True)

        return ct, ai, qc, ai_path, qc_path

    # init load
    ct0, ai0, qc0, _, _ = load_case(all_cases[0])
    ct_np = sitk.GetArrayFromImage(ct0)
    max_z, max_y = int(ct_np.shape[0]), int(ct_np.shape[1])
    scale = spacing_zyx(ct0)

    # coverage init
    if ai0 is not None and qc0 is not None:
        cov0 = coverage_ratio(ai0, qc0)
    else:
        cov0 = 0.0
    chart.update_chart(cov0)

    # contrast limits
    if normalized_01:
        img_clim = (0.0, 1.0)
        fill_value = 0.0
    else:
        # HU-like fallback
        img_clim = (np.percentile(ct_np, 1), np.percentile(ct_np, 99))
        fill_value = -1024.0

    # layers
    l_img = viewer.add_image(ct_np, name=f"CT: {all_cases[0]}", scale=scale,
                             colormap="gray", contrast_limits=img_clim)
    l_ai = viewer.add_image(
        sitk.GetArrayFromImage(ai0) if ai0 is not None else np.zeros_like(ct_np),
        name="AI mask", colormap="red", opacity=0.5, scale=scale,
        contrast_limits=(0, 1), blending="additive", visible=bool(ai0)
    )
    l_qc = viewer.add_image(
        sitk.GetArrayFromImage(qc0) if qc0 is not None else np.zeros_like(ct_np),
        name="QC mask", colormap="cyan", opacity=0.5, scale=scale,
        contrast_limits=(0, 1), blending="additive", visible=bool(qc0)
    )

    viewer.dims.order = (1, 0, 2)

    @magicgui(
        layout="vertical",
        case_key={"choices": all_cases, "label": "Case"},
        z_range={"widget_type": "RangeSlider", "min": 0, "max": max_z, "label": "Crop Z"},
        y_range={"widget_type": "RangeSlider", "min": 0, "max": max_y, "label": "Crop Y"},
        ai_opacity={"min": 0, "max": 1, "step": 0.05, "label": "AI opacity", "value": 0.5},
        qc_opacity={"min": 0, "max": 1, "step": 0.05, "label": "QC opacity", "value": 0.5},
    )
    def widget(case_key: str, z_range=(0, max_z), y_range=(0, max_y), ai_opacity=0.5, qc_opacity=0.5):
        nonlocal max_z, max_y

        # switch case
        target_name = f"CT: {case_key}"
        if l_img.name != target_name:
            ct, ai, qc, ai_path, qc_path = load_case(case_key)
            ct_np2 = sitk.GetArrayFromImage(ct)

            max_z, max_y = int(ct_np2.shape[0]), int(ct_np2.shape[1])
            widget.z_range.max = max_z
            widget.z_range.value = (0, max_z)
            widget.y_range.max = max_y
            widget.y_range.value = (0, max_y)
            widget.raw_ct = ct_np2

            l_img.name = target_name
            l_img.data = ct_np2
            l_img.scale = spacing_zyx(ct)

            l_ai.data = sitk.GetArrayFromImage(ai) if ai is not None else np.zeros_like(ct_np2)
            l_ai.visible = bool(ai)

            l_qc.data = sitk.GetArrayFromImage(qc) if qc is not None else np.zeros_like(ct_np2)
            l_qc.visible = bool(qc)

            if ai is not None and qc is not None:
                cov = coverage_ratio(ai, qc)
                chart.update_chart(cov)
                print(f"[CASE] {case_key}  coverage={cov:.3%}")
                print(f"       AI: {os.path.basename(ai_path) if ai_path else 'None'}")
                print(f"       QC: {os.path.basename(qc_path) if qc_path else 'None'}")
            else:
                chart.update_chart(0.0)
                print(f"[CASE] {case_key}  missing masks (AI or QC).")

        # live crop (only CT)
        if not hasattr(widget, "raw_ct"):
            widget.raw_ct = ct_np
        display = widget.raw_ct.copy()

        z0, z1 = int(z_range[0]), int(z_range[1])
        y0, y1 = int(y_range[0]), int(y_range[1])

        if z0 > 0:
            display[:z0, :, :] = fill_value
        if z1 < max_z:
            display[z1:, :, :] = fill_value
        if y0 > 0:
            display[:, :y0, :] = fill_value
        if y1 < max_y:
            display[:, y1:, :] = fill_value

        l_img.data = display
        l_ai.opacity = float(ai_opacity)
        l_qc.opacity = float(qc_opacity)

    viewer.window.add_dock_widget(widget, area="right")
    napari.run()


def main():
    ap = argparse.ArgumentParser(description="Napari overlay + coverage viewer (public/anonymized).")
    ap.add_argument("--img_dir", required=True, help="CT directory (NIfTI/MHA)")
    ap.add_argument("--ai_dir", required=True, help="AI mask directory")
    ap.add_argument("--qc_dir", required=True, help="QC/resection mask directory")
    ap.add_argument("--risk_threshold", type=float, default=0.72, help="Risk threshold for coverage (default: 0.72)")
    ap.add_argument("--normalized_01", action="store_true", default=False,
                    help="If set, CT is assumed in [0,1] and window is fixed to (0,1).")
    args = ap.parse_args()

    run_viewer(
        img_dir=args.img_dir,
        ai_mask_dir=args.ai_dir,
        qc_mask_dir=args.qc_dir,
        risk_threshold=args.risk_threshold,
        normalized_01=bool(args.normalized_01),
    )

if __name__ == "__main__":
    main()
