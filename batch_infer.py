#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch parallel inference script (multi-GPU).
- Auto-fix nnU-Net file naming: *.nii.gz -> *_0000.nii.gz
- Parallelize across multiple GPUs by assigning each folder to one GPU thread
- Uses nnUNetv2_predict CLI and loads checkpoints from nnUNet_results directory

Recommended weights layout inside repo:
repo_root/
  weights/nnunet_results/<experiment>/<DatasetName>/<Trainer>__<Plans>__<Config>/fold_k/<checkpoint>.pth
"""

import os
import sys
import subprocess
import threading
import queue
import time
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Optional


# ---------------------- Logging ----------------------
print_lock = threading.Lock()


def log(msg: str):
    with print_lock:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------- Utilities ----------------------
def repo_root_from_file(file_path: Path, parents_up: int = 1) -> Path:
    """
    scripts/batch_infer.py -> repo root is parents_up=1 (parent of scripts)
    adjust if you place file elsewhere.
    """
    p = file_path.resolve()
    for _ in range(parents_up):
        p = p.parent
    return p


def ensure_filenames_correct(folder_path: Path):
    """
    Ensure nnU-Net naming convention: each modality file ends with _0000.nii.gz.
    Thread-safe enough with try/except. Won't overwrite if already exists.
    """
    files = list(folder_path.glob("*.nii.gz"))
    if not files:
        return

    count_fixed = 0
    for f in files:
        name = f.name
        if name.endswith("_0000.nii.gz"):
            continue
        new_name = name.replace(".nii.gz", "_0000.nii.gz")
        new_path = f.parent / new_name

        # avoid overwriting in case both exist
        if new_path.exists():
            continue
        try:
            f.rename(new_path)
            count_fixed += 1
        except OSError:
            # Rare race-condition in multi-thread; ignore safely
            pass

    if count_fixed > 0:
        log(f"[{folder_path.name}] Auto-fixed {count_fixed} filenames to *_0000.nii.gz")


def run_cmd(cmd: str, gpu_id: int, env: dict):
    """
    Run command on specified GPU via CUDA_VISIBLE_DEVICES.
    """
    log(f"[GPU {gpu_id}] START CMD: {cmd}")
    env2 = env.copy()
    env2["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    ret = subprocess.call(cmd, shell=True, env=env2)
    if ret != 0:
        raise RuntimeError(f"Command failed on GPU {gpu_id} with exit code {ret}")


def model_base_dir(nnunet_results_exp_dir: Path, dataset_name: str, trainer: str, plans: str, config: str) -> Path:
    """
    <nnunet_results_exp_dir>/<DatasetName>/<Trainer>__<Plans>__<Config>/
    """
    return nnunet_results_exp_dir / dataset_name / f"{trainer}__{plans}__{config}"


def detect_folds(base: Path, checkpoint_name: str, max_folds: int = 5) -> List[int]:
    """
    Detect folds that contain the checkpoint file.
    """
    if not base.is_dir():
        raise FileNotFoundError(f"Model base directory not found: {base}")

    folds_found: List[int] = []
    for k in range(max_folds):
        if (base / f"fold_{k}" / checkpoint_name).is_file():
            folds_found.append(k)
    return folds_found


def build_predict_cmd(
    images_dir: Path,
    output_dir: Path,
    dataset_id: int,
    config: str,
    folds: List[int],
    trainer: str,
    plans: str,
    checkpoint_name: str,
    save_prob: bool,
    disable_tta: bool,
) -> str:
    folds_str = " ".join(str(f) for f in folds)

    cmd_parts = [
        "nnUNetv2_predict",
        f"-i \"{str(images_dir)}\"",
        f"-o \"{str(output_dir)}\"",
        f"-d {dataset_id}",
        f"-c {config}",
        f"-f {folds_str}",
        f"-tr {trainer}",
        f"-p {plans}",
        f"-chk {checkpoint_name}",
    ]
    if save_prob:
        cmd_parts.append("--save_probabilities")
    if disable_tta:
        cmd_parts.append("--disable_tta")

    return " ".join(cmd_parts)


def predict_one_dir(
    images_dir: Path,
    output_root: Path,
    folds: List[int],
    gpu_id: int,
    dataset_id: int,
    config: str,
    trainer: str,
    plans: str,
    checkpoint_name: str,
    save_prob: bool,
    disable_tta: bool,
    env: dict,
) -> Path:
    """
    Predict for a single folder.
    """
    folder_name = images_dir.name
    output_dir = output_root / f"{folder_name}_pred"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_predict_cmd(
        images_dir=images_dir,
        output_dir=output_dir,
        dataset_id=dataset_id,
        config=config,
        folds=folds,
        trainer=trainer,
        plans=plans,
        checkpoint_name=checkpoint_name,
        save_prob=save_prob,
        disable_tta=disable_tta,
    )
    run_cmd(cmd, gpu_id, env)
    return output_dir


def worker_thread(
    gpu_id: int,
    task_queue: queue.Queue,
    folds: List[int],
    args: argparse.Namespace,
    env: dict,
):
    """
    Worker that consumes folder tasks until queue is empty.
    """
    log(f"🔥 GPU {gpu_id} worker started")

    while True:
        try:
            sub_dir: Path = task_queue.get_nowait()
        except queue.Empty:
            log(f"💤 GPU {gpu_id} queue empty, worker exits")
            break

        try:
            log(f"🚀 [GPU {gpu_id}] Processing: {sub_dir.name}")

            # 1) fix filenames
            ensure_filenames_correct(sub_dir)

            # 2) run prediction
            out = predict_one_dir(
                images_dir=sub_dir,
                output_root=args.output_root,
                folds=folds,
                gpu_id=gpu_id,
                dataset_id=args.dataset_id,
                config=args.config,
                trainer=args.trainer,
                plans=args.plans,
                checkpoint_name=args.checkpoint,
                save_prob=args.save_prob,
                disable_tta=args.disable_tta,
                env=env,
            )
            log(f"✅ [GPU {gpu_id}] Done: {sub_dir.name} -> {out.name}")

        except Exception as e:
            log(f"❌ [GPU {gpu_id}] Failed: {sub_dir.name} | {e}")
        finally:
            task_queue.task_done()


# ---------------------- Main ----------------------
def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    repo_root = repo_root_from_file(here, parents_up=1)  # scripts/ -> repo root
    default_weights_root = repo_root / "weights" / "nnunet_results"

    p = argparse.ArgumentParser(
        description="Multi-GPU batch inference for nnUNetv2 with custom trainer/network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required input/output
    p.add_argument("--input_root", type=Path, required=True,
                   help="Root directory containing subfolders of NIfTI (*.nii.gz). Each subfolder is treated as a case-batch.")
    p.add_argument("--output_root", type=Path, default=repo_root / "outputs",
                   help="Output root directory. Each input subfolder produces <name>_pred/.")

    # nnU-Net identifiers
    p.add_argument("--dataset_id", type=int, default=10077, help="nnU-Net dataset ID")
    p.add_argument("--dataset_name", type=str, default="Dataset10077_MyTask", help="nnU-Net dataset name folder")
    p.add_argument("--trainer", type=str, default="nnUNetTrainerBoneAttention", help="Trainer class name")
    p.add_argument("--plans", type=str, default="nnUNetPlans", help="Plans name")
    p.add_argument("--config", type=str, default="3d_fullres", help="Configuration name")
    p.add_argument("--checkpoint", type=str, default="checkpoint_best.pth", help="Checkpoint filename")

    # Where weights live (inside repo)
    p.add_argument("--weights_root", type=Path, default=default_weights_root,
                   help="Path to repo weights root (contains nnunet_results).")
    p.add_argument("--experiment", type=str, default="all_on",
                   help="Experiment subfolder under nnUNet_results (e.g., all_on)")

    # nnU-Net environment dirs (optional; many installs still need them)
    p.add_argument("--nnunet_raw", type=Path, default=None,
                   help="Path for nnUNet_raw (optional). If provided, exported as env var.")
    p.add_argument("--nnunet_preprocessed", type=Path, default=None,
                   help="Path for nnUNet_preprocessed (optional). If provided, exported as env var.")

    # Inference toggles
    p.add_argument("--save_prob", action="store_true", default=True, help="Save probabilities")
    p.add_argument("--disable_tta", action="store_true", default=False, help="Disable test-time augmentation")

    # GPUs
    p.add_argument("--gpus", type=str, default="0,1,2",
                   help="Comma-separated GPU IDs to use, e.g. '0,1,2'")

    # Model feature switches (keep your defaults)
    p.add_argument("--bone_use_cbam", type=str, default="1")
    p.add_argument("--bone_use_aspp", type=str, default="1")
    p.add_argument("--bone_use_fpn",  type=str, default="1")
    p.add_argument("--bone_use_inmodel_norm", type=str, default="0")
    p.add_argument("--bone_use_bone_prior", type=str, default="1")
    p.add_argument("--bone_use_artifact_prior", type=str, default="1")

    # Dynamic HU thresholds / attention params
    p.add_argument("--bone_att_alpha", type=str, default="0.5")
    p.add_argument("--bone_att_loc", type=str, default="all")
    p.add_argument("--bone_att_kernel", type=str, default="3")
    p.add_argument("--bone_bone_hu_low", type=str, default="180")
    p.add_argument("--bone_prior_dynamic_high", type=str, default="1")

    return p.parse_args()


def main():
    args = parse_args()

    # Normalize paths
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.weights_root = args.weights_root.resolve()

    # Parse GPUs
    try:
        available_gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
    except ValueError:
        raise ValueError(f"Invalid --gpus format: {args.gpus}. Example: --gpus 0,1,2")

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    # Set env for nnUNet
    env = os.environ.copy()

    # nnUNet_results points to: <weights_root>/<experiment>
    nnunet_results_exp_dir = args.weights_root / args.experiment
    if not nnunet_results_exp_dir.exists():
        raise FileNotFoundError(
            f"nnUNet results directory not found: {nnunet_results_exp_dir}\n"
            f"Expected weights layout: {args.weights_root}/<experiment>/Dataset.../Trainer__Plans__Config/fold_k/{args.checkpoint}"
        )
    env["nnUNet_results"] = str(nnunet_results_exp_dir)

    # Optional: raw / preprocessed (depends on your nnUNet installation)
    if args.nnunet_raw is not None:
        env["nnUNet_raw"] = str(args.nnunet_raw.resolve())
    if args.nnunet_preprocessed is not None:
        env["nnUNet_preprocessed"] = str(args.nnunet_preprocessed.resolve())

    # Your feature switches
    env["BONE_USE_CBAM"] = args.bone_use_cbam
    env["BONE_USE_ASPP"] = args.bone_use_aspp
    env["BONE_USE_FPN"]  = args.bone_use_fpn
    env["BONE_USE_INMODEL_NORM"] = args.bone_use_inmodel_norm

    env["BONE_USE_BONE_PRIOR"] = args.bone_use_bone_prior
    env["BONE_USE_ARTIFACT_PRIOR"] = args.bone_use_artifact_prior

    env["BONE_ATT_ALPHA"] = args.bone_att_alpha
    env["BONE_ATT_LOC"] = args.bone_att_loc
    env["BONE_ATT_KERNEL"] = args.bone_att_kernel
    env["BONE_BONE_HU_LOW"] = args.bone_bone_hu_low
    env["BONE_PRIOR_DYNAMIC_HIGH"] = args.bone_prior_dynamic_high

    log("===== Multi-GPU batch inference =====")
    log(f"input_root      = {args.input_root}")
    log(f"output_root     = {args.output_root}")
    log(f"nnUNet_results  = {env.get('nnUNet_results')}")
    if "nnUNet_raw" in env:
        log(f"nnUNet_raw      = {env.get('nnUNet_raw')}")
    if "nnUNet_preprocessed" in env:
        log(f"nnUNet_preproc  = {env.get('nnUNet_preprocessed')}")
    log(f"dataset_id/name = {args.dataset_id} / {args.dataset_name}")
    log(f"trainer/plans/c = {args.trainer} / {args.plans} / {args.config}")
    log(f"checkpoint      = {args.checkpoint}")
    log(f"GPUs            = {available_gpus}")

    # Detect folds by scanning weights dir
    base = model_base_dir(nnunet_results_exp_dir, args.dataset_name, args.trainer, args.plans, args.config)
    folds = detect_folds(base, args.checkpoint, max_folds=5)
    if not folds:
        raise RuntimeError(
            f"No folds found under: {base}\n"
            f"Expected fold_k/{args.checkpoint} exists. Please check your weights layout."
        )
    log(f"Detected folds: {folds}")

    # Enumerate all subfolders under input_root
    sub_dirs = sorted([p for p in args.input_root.iterdir() if p.is_dir()])
    total_tasks = len(sub_dirs)
    if total_tasks == 0:
        raise RuntimeError(f"No sub-directories found under input_root: {args.input_root}")

    log(f"Found {total_tasks} folders to process. Enqueue tasks...")

    task_queue: queue.Queue = queue.Queue()
    for d in sub_dirs:
        task_queue.put(d)

    # Start worker threads (one per GPU)
    threads = []
    for gpu_id in available_gpus:
        t = threading.Thread(target=worker_thread, args=(gpu_id, task_queue, folds, args, env), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(1.0)  # small stagger to reduce IO burst

    # Wait for completion
    task_queue.join()
    for t in threads:
        t.join()

    log(f"===== All {total_tasks} tasks completed =====")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {e}")
        sys.exit(1)
