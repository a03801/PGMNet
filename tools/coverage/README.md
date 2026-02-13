# Napari Manual QC Viewer for CT + Mask Overlay (Public Release)

This folder contains a **Napari-based interactive viewer** for **manual quality control (QC)** of 3D medical imaging cases by overlaying:

- a reference volume (e.g., CT)
- an **AI mask** (prediction)
- a **QC / surgical / resection mask** (manual or postoperative)

The viewer computes and displays an intuitive **coverage ratio** gauge for quick human review.

> **Important:** This repository **does not** include any datasets or analysis scripts used in the paper.  
> Only the **viewer tool** is released to support visualization and manual QC.

---

## What the Viewer Shows

For each case, the viewer loads:

- **CT volume** (grayscale)
- **AI mask** overlay (red, additive)
- **QC mask** overlay (cyan, additive)
- **Coverage Ratio** displayed as a circular gauge (right panel)

### Coverage Ratio Definition (for display)
The viewer computes:

\[
\text{Coverage} = \frac{|AI \cap QC|}{|AI|}
\]

Where `AI` and `QC` are binary masks (non-zero = 1).

> Note: This ratio is displayed for QC convenience only.  
> Any statistical analysis or paper-specific modeling is **not** part of this release.

---

## Supported Input Formats

- `*.nii`, `*.nii.gz`, `*.mha`

---

## Case Matching Logic (Deterministic)

The viewer determines a **case key** from each CT filename:

1. remove extensions (`.nii`, `.nii.gz`, `.mha`)
2. remove a trailing `_0000`-style suffix if present (`_<4 digits>`)

Example:
- `005_0000.nii.gz` → case key = `005`
- `case12_scan.nii.gz` → case key = `case12_scan`

Masks are matched to the case key using:
1. **prefix match**: mask filename starts with the case key (case-insensitive)
2. fallback **substring match**: case key appears anywhere in the filename

The first deterministic match is selected.

---

## Resampling Rules

To ensure overlays align visually:

- Both masks are resampled into the **CT geometry** using **nearest neighbor interpolation**.
- CT is displayed in its native grid.

---

## Installation

### Option A — Minimal (recommended)
```bash
pip install napari[all] magicgui pyqt5 matplotlib SimpleITK numpy
````

### Option B — Conda

```bash
conda create -n napari_qc python=3.10 -y
conda activate napari_qc
pip install napari[all] magicgui pyqt5 matplotlib SimpleITK numpy
```

---

## Run

### Windows PowerShell example

```powershell
python napari_coverage_viewer.py `
  --img_dir "\\SERVER\SHARE\ct_images" `
  --ai_dir  "D:\DATA\ai_masks" `
  --qc_dir  "D:\DATA\qc_masks" `
  --normalized_01
```

### Linux/macOS example

```bash
python napari_coverage_viewer.py \
  --img_dir "/data/ct_images" \
  --ai_dir  "/data/ai_masks" \
  --qc_dir  "/data/qc_masks" \
  --normalized_01
```

---

## Command-line Arguments

* `--img_dir` (required)
  Directory containing CT volumes.

* `--ai_dir` (required)
  Directory containing AI masks.

* `--qc_dir` (required)
  Directory containing QC/surgical/resection masks.

* `--risk_threshold` (optional, default: `0.72`)
  Threshold used for the simple “low/high risk” label in the gauge.

* `--normalized_01` (optional flag)
  If set, CT is assumed to be normalized to `[0,1]` and the viewer uses a fixed window `(0,1)`.

If `--normalized_01` is NOT set, the viewer uses percentile-based contrast for CT display (1–99%).

---

## Output Files

This viewer is intended for **interactive QC** and does not write any output files by default.

If you want export functionality (screenshots / corrected masks / review logs), open an issue or extend the tool locally.

---

## Privacy & Security

Do **NOT** commit the following into a public repository:

* CT volumes or masks (`*.nii*`, `*.mha`, `*.dcm`)
* screenshots that may contain identifying anatomy or file paths
* internal network paths (UNC/IP/usernames)
* any patient-level tables

Recommended `.gitignore` entries:

```gitignore
**/*.nii
**/*.nii.gz
**/*.mha
**/*.dcm
**/*qc*.png
**/*.sav
**/*.xlsx
**/*.xls
**/*.csv
outputs/
.env
config.local.*
*.local.yaml
*.local.yml
```

---

## Folder Suggestion

Recommended layout:

```
tools/coverage/
  napari_coverage_viewer.py
  README.md
```

---

## Citation

If you use or extend this viewer in academic work, please cite the associated paper (details to be added by the authors).

---

## Troubleshooting

### “Everything is black”

Most common causes:

* CT is not normalized to `[0,1]` but `--normalized_01` was used
* contrast/window needs adjustment

Try running **without** `--normalized_01`, or verify your preprocessing.

### Masks do not appear / wrong case matched

* Check that mask filenames contain the correct case key.
* Prefer consistent naming such as:

  * `005_ai_mask.nii.gz`
  * `005_qc_mask.nii.gz`

---

```
```
