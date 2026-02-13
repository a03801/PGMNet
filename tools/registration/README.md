# Reproducible CT Registration + Manual QC (Napari)

This module defines a **reproducible** workflow for:
1) **Automatic pre/post CT registration** (Rigid → Affine, SimpleITK) using a **bone ROI** for optimization.
2) **Manual QC and correction** (Napari) that can apply **translation + rotation + scaling** as a *delta affine* **without re-running registration**, and exports artifacts so anyone can reproduce the exact correction.

> ⚠️ **Privacy / Security**
> - Do **NOT** commit any patient images (`.nii/.nii.gz/.mha/.dcm`), outputs, screenshots, or internal network paths (UNC/IP/usernames).
> - Use placeholder paths in docs and configs.

---

<p align="center">
  <img src="assets/001_1.png" width="240" />
  <img src="assets/001_2.png" width="240" />
  <img src="assets/001_3.png" width="240" />
  <img src="assets/001_4.png" width="240" />
</p>






## 0) What “Reproducible” Means Here

This workflow is reproducible because it defines a strict **Output Contract**.  
The manual QC script does **not** assume any directory naming scheme. It **discovers** valid cases by recursively scanning `out_root` and validating required files.

---

## 1) Input Requirements

### 1.1 Formats
Supported: `*.nii`, `*.nii.gz`, `*.mha`

### 1.2 Intensity Convention (Required)
All CT inputs must already be:
- clipped to a consistent HU range (example: `[200, 1800]` HU)
- normalized to **[0, 1]**

If inputs are not `[0,1]`, the bone ROI thresholding and the QC windowing will be invalid.

---

## 2) Hard-coded Bone ROI Rules (Fixed)

To ensure consistent behavior across users, the following are fixed:

- `BONE_THR = 0.19`
- `BONE_HI  = 0.90`  (**hard-coded**)

Bone ROI mask definition:

```

bone = (img > 0.19) AND (img <= 0.90)

```

`BONE_HI` excludes extremely high values (e.g., metal/saturation) from dominating the metric.

---

## 3) Step 1 — Automatic Registration (SimpleITK)

### 3.1 Output Contract (Mandatory)
For each case, the registration step produces a **case output directory** (`case_dir`).
A `case_dir` is considered **valid** only if it contains all of:

1) `registered_postop_ct.nii.gz`  
2) `final_transform.tfm`  
3) `report.json`

**Recommendation (strong):** `report.json` should include:
- `case_prefix3` (preferred)  
OR at least:
- `pre_ct` (so the case id can be deterministically derived from the pre-op filename prefix)

> The manual QC script will discover cases by enforcing the Output Contract above.

### 3.2 Example Output Layout (Example Only)
This is a *recommended* layout, but **not required** by the QC script:

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

### 3.3 Run Examples (Replace with your own paths)

**Mode A — Provide paths explicitly**
```bash
python ct_reg_bone_mi.py \
  --pre_ct  "/path/to/pre/005_0000.nii.gz" \
  --post_ct "/path/to/post/005_post.nii.gz" \
  --out_dir "/path/to/out_root/005" \
  --qc_png
````

**Mode B — Auto-match by case prefix (first 3 characters)**

```bash
python ct_reg_bone_mi.py \
  --case_id 005 \
  --pre_base  "/path/to/pre_dir" \
  --post_base "/path/to/post_dir" \
  --out_base  "/path/to/out_root" \
  --qc_png
```

---

## 4) Step 2 — Manual QC + Manual Correction Export (Napari)

The QC script loads:

* **Pre-op CT** as the fixed reference (from `--pre_dir`)
* **Auto-registered post-op CT** (`registered_postop_ct.nii.gz`) from `--out_root`

You can visually inspect overlay and apply a **delta affine**:

* Translation (mm)
* Rotation (degrees)
* Scaling (unitless)

Then you export:

* a **baked** corrected image (`registered_postop_ct_manual.nii.gz` by default)
* the **delta affine transform** used (`manual_delta_affine.tfm`)
* a **JSON record** of parameters (`manual_qc.json`)
* optional QC screenshot (`qc_manual.png`)

### 4.1 Deterministic Case Discovery (No Assumptions)

At startup the QC script:

1. recursively scans `--out_root`
2. validates Output Contract files
3. determines case id from `report.json` (preferred)
4. populates the UI case list from:

   * `prefixes_in_pre_dir ∩ prefixes_discovered_in_out_root`

### 4.2 Run (Recommended: strict mode)

```bash
python napari_manual_qc.py \
  --pre_dir  "/path/to/pre_dir" \
  --out_root "/path/to/out_root" \
  --strict
```

`--strict` = require `report.json + final_transform.tfm + registered_postop_ct.nii.gz` for every case.

### 4.3 What Gets Written (Per Case)

Outputs are saved into the corresponding `case_dir`:

* `registered_postop_ct_manual.nii.gz` *(default; unless overwrite enabled)*
* `manual_delta_affine.tfm`
* `manual_qc.json`
* `qc_manual.png` *(optional)*

**Optional overwrite:** You may overwrite `registered_postop_ct.nii.gz`, but it is recommended to keep both auto and manual outputs for auditability.

### 4.4 Units and Transform Convention (Important Detail)

* Napari translation is `(z, y, x)`
* With `scale=spacing`, translation units are **millimeters (mm)**

When exporting, the script saves a delta transform that reproduces Napari movement and writes a baked image.
In the saved JSON you will see the sign/model used, so another user can reproduce the correction exactly.

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

## 6) Repository Hygiene (Avoid Leaks)

### 6.1 Do NOT commit

* any patient images (`*.nii`, `*.nii.gz`, `*.mha`, `*.dcm`)
* any outputs (`registration_outputs/`, QC PNGs, transforms)
* any internal UNC/IP paths or usernames

### 6.2 Recommended `.gitignore`

```gitignore
**/*.nii
**/*.nii.gz
**/*.mha
**/*.dcm
**/registration_outputs/
**/qc_preview*.png
**/qc_manual*.png
**/*.tfm
**/report.json
config.local.*
.env
```

> Tip: keep `report.json` out of git if it contains sensitive paths.
> If you need reports in git, ensure they contain only sanitized, relative paths.

---

## 7) FAQ

### Q1) Napari shows everything black

Most common causes:

* input intensities are not normalized to `[0,1]`
* windowing is not appropriate
  This workflow expects `[0,1]` and typically uses percentile auto-windowing.

### Q2) Can I do rotation/scale manually?

Yes. Manual QC supports **translation + rotation + scaling** and exports a **delta affine transform** plus a baked corrected image.

### Q3) Why strict mode?

Strict mode prevents any ambiguous “guessing” of where outputs are.
It guarantees deterministic discovery via the Output Contract.

---

## 8) Suggested Repo Layout

Recommended:

```
tools/registration/
  ct_reg_bone_mi.py
  napari_manual_qc.py
  README.md
```

```
```
