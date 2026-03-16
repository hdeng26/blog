---
layout: post
title:  "PyTorch GPU Setup with Conda (2026 Simplified Edition)"
date:   2026-03-16 12:00:00 +0530
---

**This is an updated version of my [2024 PyTorch Windows environment config tutorial]({% post_url 2024-01-22-torch_gpu %}). The old 9-step process has been simplified to 5 steps — no more manual CUDA Toolkit, cuDNN, or environment variable configuration.**

---

## What Changed?

In 2024, setting up PyTorch with GPU support meant manually installing the CUDA Toolkit, downloading cuDNN from NVIDIA's developer site, copying library files into system directories, and configuring environment variables. That was 9 steps with plenty of room for version mismatches.

Today, **none of that is necessary**. Modern PyTorch pip wheels bundle their own CUDA runtime and cuDNN libraries directly inside the package. The only system-level requirement is the NVIDIA GPU driver, which most people already have.

Also worth noting: **PyTorch officially deprecated their Anaconda channel after version 2.5** (October 2024). The old `conda install pytorch ... -c pytorch -c nvidia` command no longer works for current releases. The recommended approach is now `conda` for environment management + `pip` for the actual PyTorch install.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step 1: Install Miniconda](#step-1-install-miniconda)
3. [Step 2: Create a New Conda Environment](#step-2-create-a-new-conda-environment)
4. [Step 3: Install PyTorch with GPU Support](#step-3-install-pytorch-with-gpu-support)
5. [Step 4: Install Additional Packages](#step-4-install-additional-packages)
6. [Step 5: Verify GPU Access](#step-5-verify-gpu-access)
7. [CPU-Only Installation](#cpu-only-installation)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

You need exactly **one** thing installed at the system level: the **NVIDIA GPU Driver**.

Check if you already have it:

```bash
nvidia-smi
```

If this command returns a table showing your GPU name and driver version, you are good to go — skip to Step 1. If not:

- **Windows:** Download from [https://www.nvidia.com/drivers](https://www.nvidia.com/drivers), or let Windows Update / GeForce Experience handle it.
- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt update
  sudo apt install nvidia-driver-565
  ```
  Replace `565` with the latest recommended version for your GPU. Reboot after installation.

**Take note of the CUDA Version** shown in the top-right corner of `nvidia-smi` output (e.g., `CUDA Version: 12.8`). This is the **maximum** CUDA version your driver supports. You will need this in Step 3.

That's it. No CUDA Toolkit download, no cuDNN, no environment variables.

---

## Step 1 Install Miniconda

We use **Miniconda** instead of full Anaconda — it is smaller, faster, and provides the same `conda` command for environment management.

### Windows

Download the installer from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html) and run it. After installation, open **Anaconda Prompt** (search in Start Menu) for all subsequent commands.

### Linux

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the prompts and say "yes" to initialize conda. Then restart your terminal or run `source ~/.bashrc`.

### Verify

```bash
conda --version
```

---

## Step 2 Create a New Conda Environment

**Why not just use `base`?** Your base environment (or system Python) may have packages that conflict with PyTorch. A dedicated environment keeps everything isolated and reproducible. If something breaks, you can delete it and start over without affecting anything else.

### Choose the right Python version

**Use Python 3.12.** PyTorch stable requires Python >= 3.10, and 3.12 is currently the most stable choice for the deep learning ecosystem. Python 3.13 and 3.14 are supported by PyTorch itself but some third-party packages (e.g., certain versions of `scikit-learn`, `transformers`, or `opencv`) may not have caught up yet.

**Rule of thumb:** use the second-latest stable Python release for deep learning work.

```bash
conda create -n torch python=3.12 -y
```

Activate the environment:

```bash
conda activate torch
```

Your terminal prompt should now show `(torch)` at the beginning. **All following commands assume this environment is active.**

---

## Step 3 Install PyTorch with GPU Support

> **Note:** The old `conda install pytorch ... -c pytorch -c nvidia` method is deprecated. PyTorch stopped publishing to their official Anaconda channel after v2.5. Use `pip` inside your conda environment instead.

Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) to get the exact command for your setup. For most users with a recent NVIDIA GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Which CUDA version to choose?

The CUDA version in the `--index-url` must be **equal to or lower than** the CUDA version shown by `nvidia-smi`:

| Your `nvidia-smi` CUDA Version | Recommended `--index-url` suffix |
|-------------------------------|----------------------------------|
| 12.8 or higher                | `cu128`                          |
| 12.6 – 12.7                  | `cu126`                          |
| 11.8 – 12.5                  | `cu118`                          |

### What does this command actually do?

The pip wheel you install includes **everything** — CUDA runtime, cuDNN, and all supporting libraries — bundled inside your conda environment. This is why Steps 3, 4, and 5 from the old tutorial (CUDA Toolkit, cuDNN, environment variables) are no longer needed.

---

## Step 4 Install Additional Packages

With PyTorch installed via pip, keep other packages on pip too to avoid dependency conflicts:

```bash
# Common data science and ML packages
pip install numpy pandas matplotlib scikit-learn jupyter

# NLP (Hugging Face)
pip install transformers datasets tokenizers accelerate

# Computer vision
pip install opencv-python pillow
```

> **Tip:** Avoid mixing `conda install` and `pip install` for packages that depend on PyTorch. Since PyTorch itself came from pip, its dependents should also come from pip.

---

## Step 5 Verify GPU Access

Run this in Python to confirm everything works:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version:    {torch.version.cuda}")
    print(f"cuDNN version:   {torch.backends.cudnn.version()}")
    print(f"GPU device:      {torch.cuda.get_device_name(0)}")
    print(f"GPU count:       {torch.cuda.device_count()}")

    # Quick computation test on GPU
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")
    z = x @ y
    print(f"\nGPU computation test passed!")
    print(f"Result shape: {z.shape}, device: {z.device}")
else:
    print("No GPU detected. Running in CPU mode.")
```

**Expected output:**

```
PyTorch version: 2.7.0+cu128
CUDA available:  True
CUDA version:    12.8
cuDNN version:   90100
GPU device:      NVIDIA GeForce RTX 4070
GPU count:       1

GPU computation test passed!
Result shape: torch.Size([1000, 1000]), device: cuda:0
```

If `CUDA available` shows `True`, congratulations — your environment is ready.

---

## CPU-Only Installation

If you do not have an NVIDIA GPU, the setup is even simpler. Follow Steps 1 and 2 above, then:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

No driver installation needed.

---

## Troubleshooting

### `torch.cuda.is_available()` returns `False`

1. **Check your driver:** Run `nvidia-smi`. If it fails, your GPU driver is not installed or not loaded properly. On Linux, try rebooting.
2. **Version mismatch:** Your driver's CUDA version (from `nvidia-smi`) must be >= the PyTorch CUDA version (from `torch.version.cuda`). If not, either update your driver or reinstall PyTorch with a lower CUDA variant.
3. **Wrong wheel installed:** Run `pip show torch` and check the version string. If it says `2.7.0+cpu`, you installed the CPU build by mistake. Reinstall with the correct `--index-url`.

### Reinstalling PyTorch

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Starting completely over

```bash
conda deactivate
conda remove -n torch --all -y
conda create -n torch python=3.12 -y
conda activate torch
# Then redo Step 3
```

---

## Quick Reference

```bash
# One-time setup
conda create -n torch python=3.12 -y
conda activate torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Daily usage
conda activate torch
python train.py
conda deactivate

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Comparison: 2024 vs 2026

| | 2024 (Old Tutorial) | 2026 (This Tutorial) |
|---|---|---|
| Steps | 9 | 5 |
| CUDA Toolkit install | Manual download + install | Not needed (bundled in pip wheel) |
| cuDNN install | Manual download + copy files | Not needed (bundled in pip wheel) |
| Environment variables | Manual PATH/CUDA_PATH setup | Not needed |
| PyTorch install method | `conda install -c pytorch -c nvidia` | `pip install --index-url` |
| NVIDIA developer account | Required (for cuDNN) | Not required |
| Time to set up | ~30-60 minutes | ~5 minutes |

---

Please feel free to leave a comment below if you have any questions, suggestions, or feedback. Your input is greatly appreciated!
