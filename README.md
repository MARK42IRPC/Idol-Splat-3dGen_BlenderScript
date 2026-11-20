# IS3D - Idol Splat 3D Generator

**English** | [简体中文](README.zh-CN.md)

Turn **one reference image** into a Gaussian splat, a textured high-poly mesh, or a
UV-unwrapped low-poly mesh with baked base colour and normal maps -- on the GPU,
straight from the NVIDIA driver.

No PyTorch. No ONNX Runtime. No CUDA Toolkit. No Python packages. The inference
engine is a single 1.2 MB DLL that talks to `nvcuda.dll` (the driver that is
already installed) and JIT-compiles its kernels from embedded PTX at load time.

```
one reference image
  -> BiRefNet matte (optional)
  -> DINOv3 ViT-H/16+ image tokens + FLUX.2 VAE latents
  -> 50-step rectified-flow sampler (fp16, CFG)
  -> octree anchors -> 32 Gaussians per anchor
  -> Gaussian PLY / high-poly / low-poly + baked maps
  -> appended to the Blender scene at the 3D cursor
```

## Features

- **One image in, three kinds of asset out**: `高斯点云` (GPUSplat point cloud),
  `高模` (dense mesh with Gaussian colour transfer), `低模` (auto UV unwrap with
  baked base colour and normal maps).
- **Bundled background removal** -- BiRefNet Swin-L + deformable ASPP ported to
  the native engine, driven by the `抠除背景` checkbox. Alpha matches the PyTorch
  reference at cosine 0.99997.
- **Light footprint** -- 1.99 GiB peak VRAM with matting, 1.59 GiB without.
  Each stage frees its tensors (and its VRAM) before the next one starts, so the
  3.5 GB weight file never sits in VRAM all at once.
- **No Python dependency stack** -- nothing to `pip install`, no CUDA toolkit, no
  redistributables. If the machine has an NVIDIA driver, it runs.
- **Non-blocking UI** -- generation runs in a separate headless Blender process;
  the viewport stays responsive and the job can be cancelled.

## Requirements

| Item | Requirement |
| --- | --- |
| OS | Windows 10/11 x64 |
| GPU | NVIDIA RTX 30/40/50 series (compute capability >= 8.6) |
| Driver | any driver that JITs PTX `compute_86` (2021 or newer) |
| Blender | 5.3 or newer (Steam build works) |
| VRAM | ~2 GB used, 4 GB recommended, hard limit 6 GB |
| Disk | ~4 GB for the weights |

## Install

Pick the mirror you can reach; all three carry identical, checksummed assets.

```powershell
# GitHub
git clone https://github.com/MARK42IRPC/is3D-BlenderScript
cd is3D-BlenderScript
powershell -ExecutionPolicy Bypass -File install/install.ps1

# Hugging Face (hf-mirror for mainland China)
powershell -ExecutionPolicy Bypass -File install/install.ps1 -Mirror hf

# ModelScope (魔搭)
powershell -ExecutionPolicy Bypass -File install/install.ps1 -Mirror modelscope
```

The script finds Blender (Steam libraries and the official installer are both
handled), drops the add-on into `scripts/addons/is3d`, downloads the engine DLL
and the weight slices, checks every SHA-256, and rebuilds `idol_weights.bin`.
Then enable **IS3D** in `Edit > Preferences > Add-ons`.

| Switch | Meaning |
| --- | --- |
| `-Mirror github\|hf\|modelscope` | where the assets come from (default `github`) |
| `-AssetBaseUrl <url>` | use any other mirror, e.g. an internal file server |
| `-BlenderRoot <dir>` | point at a specific Blender install |
| `-AddonsDir <dir>` | write straight into a `scripts/addons` directory |
| `-Version 0.4.0` | install a different release |
| `-SkipAssets` | code only (useful if `is3d/native/` is already populated) |
| `-Source` | copy `is3d/` from this checkout instead of the release zip |
| `-FromZip <path>` | install an add-on zip you already have |
| `-KeepParts` | keep the downloaded weight slices in `%TEMP%` |
| `-Force` | re-download and overwrite existing files |

## Use

Open the 3D viewport sidebar (`N`) and pick the **IS3D** tab.

| Control | Meaning |
| --- | --- |
| 参考图 | the reference image |
| 抠除背景 | run the bundled BiRefNet matte first; turn off for images that already have a clean silhouette (saves ~0.4 GiB VRAM and a second) |
| 产出 | `高斯点云` (GPUSplat), `高模` (dense mesh with vertex colours), `低模` (UV unwrap + baked base colour/normal) |
| 高斯点数量 / 三角面数 | output budget |
| 输出目录 | where the job folder, PLY and reports are written |

The result is appended to the current scene at the 3D cursor as a
`IS3D 生成资产` collection, so it can be moved or deleted as a unit.

## How it works

`docs/ARCHITECTURE.md` has the stage map, the kernel inventory and the memory
strategy. In short:

1. **preprocess** -- resize, optional BiRefNet matte, subject-bbox crop.
2. **encode** -- DINOv3 ViT-H/16+ image tokens and FLUX.2 VAE latents.
3. **denoise** -- 50-step rectified-flow sampler, fp16, classifier-free guidance.
4. **anchors / gaussians** -- anchors on a predicted octree, 32 Gaussians each.
5. **export** -- Gaussian PLY + camera JSON; high/low-poly reconstruction runs
   on Blender side with the same Gaussian colours.

The engine keeps a single CUDA context, one arena per stage, and unloads each
stage's weights from VRAM as soon as it finishes.

## Model assets

| File | Size | Purpose |
| --- | --- | --- |
| `idol_engine.dll` | 1.2 MB | native inference engine (PTX embedded) |
| `idol_weights.bin` | 3.30 GiB | all stage weights, `IDOLW1` container |

Because GitHub caps release assets at 2 GiB, the weight file ships as
`idol_weights.bin.partNN` slices; the installer concatenates and re-verifies
them. Format and download order: `docs/ASSETS.md`.

Mirrors:

- GitHub: <https://github.com/MARK42IRPC/is3D-BlenderScript/releases/tag/v0.4.0>
- Hugging Face: <https://huggingface.co/Mark42IRP/is3D-BlenderScript>
- ModelScope: <https://www.modelscope.cn/models/Mark42IRPC/is3D-BlenderScript>

## Build from source

`docs/BUILD.md` covers the MSVC + CUDA 12.6 toolchain, the NVRTC path (no
`nvcc` needed), how `idol_weights.bin` is packed from the upstream checkpoints,
and how the release payload is produced.

## Validation

`validation/` holds the checks used during development:

- `check_birefnet.py` -- layer-by-layer comparison of the native BiRefNet port
  against the PyTorch reference (alpha cosine 0.99997, worst layer 0.975).
- `run_matte.py` -- end-to-end engine matte run (IoU 0.9887 vs the reference).
- `blender_package_check.py` -- installs the packaged add-on in headless Blender,
  generates an asset, imports it and renders two views, asserting coverage and
  colour spread.

## Licence

Code: MIT (see LICENSE). Model weights: see NOTICE.md -- they carry the licences
of the upstream models and are distributed for research and personal use.
