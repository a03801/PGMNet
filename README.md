
# PGMNet — Osteomyelitis CT Segmentation (NIfTI)

This repository provides the implementation of **PGMNet** (a custom **nnU-Net v2** trainer/network) and **pretrained weights** for **inference** on CT **NIfTI** files (`.nii.gz`). Inference is executed via the official `nnUNetv2_predict` CLI.

> **Scope**: inference only (pretrained models + reproducible inference script).
> ## Related tools (optional)

This repository focuses on **inference** with pretrained nnU-Net v2 weights.

If you also need **CT registration / manual QC visualization** for aligning pre/post CT or reviewing overlays, see:
- `tools/registration/` (SimpleITK registration + Napari QC viewer)

> Note: Paper-specific statistical analysis (e.g., recurrence modeling) is **not** included in this public release.

> **Input**: CT in NIfTI format (`.nii.gz`).  
> **Output**: segmentation masks (and optional probability maps).

---

## Repository structure

Key files/folders:

```

PGMNet/
BoneAttentionUNetV2.py
nnUNetTrainerBoneAttention.py
scripts/
batch_infer.py
weights/
nnunet_results/
all_on/
Dataset10077_MyTask/
nnUNetTrainerBoneAttention__nnUNetPlans__3d_fullres/
dataset.json
dataset_fingerprint.json
plans.json
fold_0/checkpoint_best.pth
fold_1/checkpoint_best.pth
fold_2/checkpoint_best.pth
fold_3/checkpoint_best.pth
fold_4/checkpoint_best.pth

````

**Important**: nnU-Net v2 requires `dataset.json` (and typically `plans.json` / fingerprint) to be present in the model folder for inference. This repo includes these files.

---

## Tested environment

The code and inference pipeline were tested on:

- **OS**: Linux
- **Python**: 3.9.23
- **PyTorch**: 2.5.1+cu121
- **CUDA (PyTorch build)**: 12.1
- **nnU-Net v2**: 2.5.2
- **GPU**: NVIDIA GeForce RTX 4070 (multi-GPU supported)

> Note: Your NVIDIA driver CUDA version can be higher than the PyTorch CUDA build (e.g., driver shows CUDA 12.8 while PyTorch is cu121). This is normal as long as the driver supports the CUDA runtime required by PyTorch.

---

## Quickstart (inference)

### 0) Clone (with Git LFS if weights are stored via LFS)

If the repository uses Git LFS for `.pth` files, install LFS and pull weights:

```bash
git lfs install
git clone https://github.com/a03801/PGMNet.git
cd PGMNet
git lfs pull
````

If you do not use Git LFS, normal `git clone` is enough.

---

## Installation

### 1) Create and activate an environment (recommended)

```bash
conda create -n nnunet python=3.9 -y
conda activate nnunet
```

### 2) Install PyTorch (GPU)

Install PyTorch that matches your CUDA runtime. For example (CUDA 12.1 build):

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

If you already have a working PyTorch + CUDA setup, you can skip this.

### 3) Install nnU-Net v2 and other dependencies

```bash
pip install nnunetv2==2.5.2
pip install -r requirements.txt
```

> If `requirements.txt` already includes `nnunetv2`, you can remove the separate `pip install nnunetv2==2.5.2`.

---

## IMPORTANT: allow nnU-Net to import the custom Trainer/Network

nnU-Net v2 must be able to import the custom trainer and network defined in this repository.
Before running inference, set `PYTHONPATH` to the repository root:

```bash
cd /path/to/PGMNet
export PYTHONPATH=$(pwd):$PYTHONPATH
```

Verify imports:

```bash
python -c "import nnUNetTrainerBoneAttention; import BoneAttentionUNetV2; print('import ok')"
```

> You may see warnings about `nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results` when importing nnU-Net.
> These warnings are common and **do not prevent inference**, because `scripts/batch_infer.py` sets `nnUNet_results` internally for the prediction call.

---

## Pretrained weights (included)

Weights are stored under `weights/nnunet_results/` in the standard nnU-Net v2 results layout:

* experiment: `all_on`
* dataset_name: `Dataset10077_MyTask`
* trainer: `nnUNetTrainerBoneAttention`
* plans: `nnUNetPlans`
* config: `3d_fullres`
* checkpoint: `checkpoint_best.pth`

---

## Input format

* Input must be **NIfTI**: `*.nii.gz`
* The script processes **subfolders** under `--input_root`
  (each subfolder is treated as one inference batch/case-group).
* nnU-Net expects modality suffix `_0000.nii.gz`.
  The script automatically renames `*.nii.gz` to `*_0000.nii.gz` if needed.

Example input structure:

```
/path/to/input_root/
  caseA/
    caseA.nii.gz
  caseB/
    caseB.nii.gz
```

---

## Run inference (multi-GPU batch)

### Multi-GPU example

```bash
cd /path/to/PGMNet
export PYTHONPATH=$(pwd):$PYTHONPATH

python scripts/batch_infer.py \
  --input_root /path/to/input_root \
  --output_root ./outputs \
  --gpus 0,1,2 \
  --weights_root ./weights/nnunet_results \
  --experiment all_on
```

### Single GPU example

```bash
cd /path/to/PGMNet
export PYTHONPATH=$(pwd):$PYTHONPATH

python scripts/batch_infer.py \
  --input_root /path/to/input_root \
  --output_root ./outputs \
  --gpus 0 \
  --weights_root ./weights/nnunet_results \
  --experiment all_on
```

Outputs will be written to:

```
outputs/
  caseA_pred/
  caseB_pred/
```

---

## Output files

* Predicted segmentation masks are written by `nnUNetv2_predict` to each `*_pred/` folder.
* If `--save_prob` is enabled (default in the script), probability maps will also be saved.

---

## Script options

To view all supported options:

```bash
python scripts/batch_infer.py -h
```

Common options include:

* `--gpus 0,1,2` (comma-separated GPU IDs)
* `--save_prob` / `--disable_tta`
* `--dataset_id`, `--dataset_name`, `--trainer`, `--plans`, `--config`, `--checkpoint`
* `--weights_root` and `--experiment` (recommended to keep default repo layout)

---

## Troubleshooting

### 1) `ModuleNotFoundError: nnUNetTrainerBoneAttention`

You likely forgot to set `PYTHONPATH`:

```bash
cd /path/to/PGMNet
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### 2) `No folds found ... checkpoint_best.pth`

Check your weights directory layout matches:

`weights/nnunet_results/all_on/.../fold_k/checkpoint_best.pth`

Also confirm `git lfs pull` was executed if using Git LFS.

### 3) `FileNotFoundError: ... dataset.json`

Your model folder is missing `dataset.json` (and possibly `plans.json`).
This repo should include them under:

`weights/nnunet_results/all_on/Dataset10077_MyTask/nnUNetTrainerBoneAttention__nnUNetPlans__3d_fullres/`

### 4) CUDA not available

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

---

## nnU-Net citation

Please cite the nnU-Net paper when using nnU-Net:

Isensee, F. et al. *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* **Nature Methods** 18, 203–211 (2021).

---

## License

See `LICENSE`.

---

## Citation (PGMNet)

If you use this repository, please cite the associated manuscript (to be updated after acceptance).

````

---




