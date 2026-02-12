# PGMNet
# PGMNet (nnU-Net v2) — Osteomyelitis CT Segmentation (NIfTI)

This repository provides the implementation of the PGMNet architecture (custom nnU-Net v2 trainer/network) and pretrained weights for **inference** on CT **NIfTI** files (`.nii.gz`). Inference is executed via `nnUNetv2_predict`.

---

## Tested environment

The code and inference pipeline have been tested on:

- **Python**: 3.9.23  
- **PyTorch**: 2.5.1+cu121  
- **CUDA (PyTorch build)**: 12.1  
- **nnU-Net v2**: 2.5.2  
- **GPU**: NVIDIA GeForce RTX 4070 (multiple GPUs)

> Note: Your NVIDIA driver CUDA version can be higher (e.g., CUDA 12.8) while PyTorch is built with CUDA 12.1; this is normal as long as the driver supports the CUDA runtime required by PyTorch.

---

## Installation

### 1) Create and activate an environment (recommended)

```bash
conda create -n nnunet python=3.9 -y
conda activate nnunet
2) Install dependencies
pip install -r requirements.txt
pip install nnunetv2==2.5.2
If you already installed nnU-Net v2, you can skip the second command.

Make nnU-Net able to import the custom Trainer/Network (IMPORTANT)
nnU-Net v2 needs to import the custom trainer and network in this repository.
Before running inference, set PYTHONPATH to the repository root:

export PYTHONPATH=/path/to/PGMNet:$PYTHONPATH
Verify imports:

python -c "import nnUNetTrainerBoneAttention; import BoneAttentionUNetV2; print('import ok')"
Pretrained weights (stored in this repository)
Weights are stored under weights/nnunet_results/ using the nnU-Net v2 results layout:

weights/
  nnunet_results/
    all_on/
      Dataset10077_MyTask/
        nnUNetTrainerBoneAttention__nnUNetPlans__3d_fullres/
          fold_0/checkpoint_best.pth
          fold_1/checkpoint_best.pth
          fold_2/checkpoint_best.pth
          fold_3/checkpoint_best.pth
          fold_4/checkpoint_best.pth
experiment: all_on

dataset_name: Dataset10077_MyTask

trainer: nnUNetTrainerBoneAttention

plans: nnUNetPlans

config: 3d_fullres

checkpoint: checkpoint_best.pth (default)

Input format
Input must be NIfTI: *.nii.gz

The script processes subfolders under --input_root. Each subfolder can contain one or multiple NIfTI files.

nnU-Net expects modality suffix _0000.nii.gz. The script will automatically rename *.nii.gz to *_0000.nii.gz if needed.

Example input structure:

/path/to/input_root/
  caseA/
    caseA.nii.gz
  caseB/
    caseB.nii.gz
Run inference (multi-GPU batch)
We provide a multi-GPU batch inference script:

python scripts/batch_infer.py \
  --input_root /path/to/input_root \
  --output_root ./outputs \
  --gpus 0,1,2
Single GPU example:

python scripts/batch_infer.py \
  --input_root /path/to/input_root \
  --output_root ./outputs \
  --gpus 0
Outputs will be written to:

outputs/
  caseA_pred/
  caseB_pred/
Troubleshooting
1) ModuleNotFoundError: nnUNetTrainerBoneAttention
You likely forgot to set PYTHONPATH:

export PYTHONPATH=/path/to/PGMNet:$PYTHONPATH
2) No folds found ... checkpoint_best.pth
Check your weights directory layout matches the expected nnU-Net v2 structure under:
weights/nnunet_results/all_on/.../fold_k/checkpoint_best.pth

3) CUDA not available
Confirm your PyTorch CUDA build and GPU availability:

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
(Recommended) Git LFS for weights
Model weights (*.pth) are typically large. We strongly recommend Git LFS:

git lfs install
git lfs track "*.pth"
git add .gitattributes
Then commit and push as usual.

License
See LICENSE.

Citation
If you use this repository, please cite the associated manuscript (to be updated after acceptance).


---

### 你还需要补 2 个“仓库层面”的东西（很重要）
1) **`.gitattributes`**（Git LFS track 生成的文件）要一起提交，不然 `.pth` 还是会按普通 Git 推。  
2) 你 README 里 `/path/to/PGMNet`、`/path/to/input_root` 这种路径示例没问题，但你要确保仓库里真的有：
- `scripts/batch_infer.py`
- `weights/nnunet_results/.../checkpoint_best.pth`（至少 fold_0–fold_4）

如果你把你仓库当前的 `weights/` 目录树（执行一次 `find weights -maxdepth 6 -type f | head` 的输出）贴出来，我可以帮你确认 README 里的“权重路径”是否完全对得上你脚本的探测逻辑，避免别人一跑就报 “No folds found”。
::contentReference[oaicite:0]{index=0}
