
# CT Registration + Manual QC (Reproducible Workflow)

This folder provides a **reproducible** two-step workflow for registering pre-op and post-op CT volumes and then performing **human-in-the-loop quality control (QC)** in Napari **without re-running registration**.

- **Step 1 — Automatic registration (SimpleITK):** Rigid → Affine, optimized using **Mattes Mutual Information** computed only within a **bone ROI**.
- **Step 2 — Manual QC (Napari):** Load the automatic output and visually inspect overlay. If needed, apply a small **translation correction (mm)** and **export** a “baked” corrected image + transform for reproducibility.

> **Privacy note:** Do not commit real patient data, screenshots, or any internal network paths (UNC/IP/usernames) to a public repository.

---

## 1) Input Assumptions (Required for Reproducibility)

### 1.1 File formats
Supported image formats:
- `*.nii`, `*.nii.gz`, `*.mha`

### 1.2 Intensity convention (critical)
All CT volumes used by this pipeline are assumed to be:
1. **Clipped** to a consistent HU window (example: HU `[200, 1800]`)
2. **Normalized** to **[0, 1]**

If your inputs are not normalized to `[0,1]`, the bone ROI definition below will be invalid and registration quality will suffer.

---

## 2) Fixed Parameters (Hard-coded)

To ensure consistent behavior across machines/users, these ROI parameters are fixed:

- `BONE_THR = 0.19`
- `BONE_HI  = 0.90`  (**hard-coded**)

Bone ROI mask is computed as:

```

bone = (img > 0.19) AND (img <= 0.90)

```

`BONE_HI` excludes very high values (e.g., metal/saturation) from the ROI so they do not dominate MI.

---

## 3) Step 1 — Automatic Registration Script

### 3.1 Output Contract (the key to reproducibility)

For each case, the registration script writes a **case output directory** (`case_dir`).  
A `case_dir` is considered a **valid** registration output **only if** it contains all of the following:

1. `registered_postop_ct.nii.gz`
2. `final_transform.tfm`
3. `report.json`  
   - strongly recommended fields:
     - `case_prefix3` (preferred)  
     - or at least `pre_ct` (so case can be deterministically inferred from the pre-op filename prefix)

> The Napari QC script will **not assume** any directory naming convention. It will **discover** valid cases by recursively scanning `out_root` and enforcing this Output Contract.

### 3.2 Recommended output layout (example only)
You may store outputs like this (but the QC script does not rely on the folder names):

```

out_root/
005/
registered_postop_ct.nii.gz
final_transform.tfm
report.json
fixed_bone_mask.nii.gz
moving_bone_mask.nii.gz
qc_preview.png
006/
...

````

### 3.3 Run examples

> Replace all paths below with your own local/server paths. Do not commit real paths into the repo.

**Mode A — Explicit paths**
```bash
python ct_reg_bone_mi.py \
  --pre_ct  "/path/to/pre/005_0000.nii.gz" \
  --post_ct "/path/to/post/005_post.nii.gz" \
  --out_dir "/path/to/out_root/005" \
  --qc_png
````

**Mode B — Case prefix auto-match (first 3 characters)**

```bash
python ct_reg_bone_mi.py \
  --case_id 005 \
  --pre_base  "/path/to/pre_dir" \
  --post_base "/path/to/post_dir" \
  --out_base  "/path/to/out_root" \
  --qc_png
```

---

## 4) Step 2 — Manual QC in Napari (No Re-registration)

The manual QC script is designed for:

* loading **pre-op CT** (fixed)
* loading the auto-registered **registered_postop_ct.nii.gz** (moving, already in fixed space)
* visually checking alignment in Napari via overlay (gray + red)
* optionally applying a small **translation correction (mm)**
* exporting corrected outputs so another user can reproduce the exact correction

### 4.1 Deterministic discovery of cases (no assumptions)

At startup, the QC script:

1. Recursively scans `--out_root` for candidate case directories.
2. Marks a case as **valid** only if it contains:

   * `registered_postop_ct.nii.gz`
   * `final_transform.tfm`
   * `report.json` (**required in strict mode**)
3. Determines `case_prefix3` primarily from `report.json`:

   * prefer `case_prefix3`, otherwise fall back to `pre_ct` filename prefix
4. Builds the QC dropdown list as:

   * `prefixes_in_pre_dir ∩ prefixes_in_out_root_index`

Therefore, given the same inputs on disk, the case list and file mapping are reproducible.

### 4.2 Run (recommended with strict validation)

```bash
python napari_manual_qc.py \
  --pre_dir  "/path/to/pre_dir" \
  --out_root "/path/to/out_root" \
  --strict
```

* `--strict` enforces the Output Contract with `report.json` and avoids any “guessing”.

### 4.3 Manual correction outputs

After adjusting translation in Napari and clicking **Save**, the script writes to the corresponding `case_dir`:

* `registered_postop_ct_manual.nii.gz` (manual-corrected image)
* `manual_delta_translation.tfm` (manual translation delta)
* `manual_qc.json` (records translation values and output paths)
* `qc_manual.png` (optional QC screenshot)

Optionally, you can choose to **overwrite** `registered_postop_ct.nii.gz` (not recommended by default—keep both versions for auditability).

### 4.4 Translation units and sign convention (reproducibility detail)

* Napari translation uses `(z, y, x)`.
* With `scale=spacing`, the translation units are **millimeters (mm)**.
* When exporting, the script converts Napari translation into a SimpleITK transform used for resampling:

  * it saves `Translation(-shift_xyz)` so that the written image is “baked” and loads aligned **without needing Napari translate**.

---

## 5) Installation

### 5.1 Automatic registration dependencies

```bash
pip install SimpleITK numpy
```

### 5.2 Napari QC dependencies

```bash
pip install napari[all] magicgui pyqt5 matplotlib SimpleITK numpy
```

---

## 6) Repository Hygiene (Do Not Leak Data)

### 6.1 Do not commit

* patient images (`*.nii*`, `*.mha`, `*.dcm`)
* any output folders (e.g., `registration_outputs/`)
* any QC PNGs if they might contain identifying overlays or paths
* internal network paths / UNC / IP addresses / usernames

### 6.2 Recommended `.gitignore`

```gitignore
**/*.nii
**/*.nii.gz
**/*.mha
**/*.dcm
**/registration_outputs/
**/qc_preview*.png
**/qc_manual*.png
config.local.*
.env
```

---

## 7) FAQ

### Q1: Why is everything black in Napari?

Most common causes:

* intensity is not normalized to `[0,1]`
* window/level is not appropriate
  This QC script typically uses percentile-based auto-windowing (1–99%). If it is still black, verify input normalization.

### Q2: Can manual QC do rotation/scale?

No. Manual QC here is translation-only, intended for small residual misalignment. For rotation/scale issues, re-run or improve the automatic registration step.

### Q3: Why use `--strict`?

Strict mode enforces the Output Contract and prevents ambiguous mapping from folder names—this is the easiest way to guarantee deterministic, reproducible loading.

---

## 8) Suggested Repo Layout

Recommended structure:

```
tools/registration/
  ct_reg_bone_mi.py
  napari_manual_qc.py
  README.md
```

```
```
