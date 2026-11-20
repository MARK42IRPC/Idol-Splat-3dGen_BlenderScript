# IS3D - Idol Splat 3D Generator

A Blender add-on that turns **one image** into a Gaussian splat, a textured
high-poly mesh, or a UV-unwrapped low-poly mesh with baked base colour and
normal maps -- on the GPU, straight from the NVIDIA driver.

No PyTorch. No ONNX Runtime. No CUDA Toolkit. No Python packages to install.
The inference engine is a single 1.2 MB DLL that talks to `nvcuda.dll` (the
driver that is already on the machine) and JIT-compiles its kernels from
embedded PTX at load time.

```
one reference image ──▶ BiRefNet matte ──▶ DINOv3 + VAE encode ──▶ flow matching
   ──▶ octree Gaussian anchors ──▶ Gaussian PLY ──▶ Blender (GPUSplat)
   ──▶ optional high/low poly reconstruction with baked textures
```

## Requirements

- Windows 10/11 x64
- NVIDIA RTX 30/40/50 series (compute capability >= 8.6; the kernels are PTX
  `compute_86`, JIT-compiled for the installed GPU)
- Blender 5.3 or newer (Steam build works)
- ~4 GB of free VRAM. Measured peak on an RTX 5060 Ti 16 GB:
  - splat, background matte on: **1.99 GiB**
  - splat, matte off: **1.59 GiB**
- Disk: ~4 GB for the weight file

## Install

```powershell
git clone https://github.com/MARK42IRPC/is3d
cd is3d
powershell -ExecutionPolicy Bypass -File install/install.ps1
```

The script finds your Blender installation (Steam and the official installer are
both handled), downloads the engine DLL plus the weight file from the GitHub
release, verifies the SHA-256 of every part, and unpacks the add-on into
`scripts/addons/is3d`. Then enable **IS3D** in `Edit > Preferences > Add-ons`.

Useful switches:

| Switch | Meaning |
| --- | --- |
| `-BlenderRoot <dir>` | point at a specific Blender install |
| `-AddonsDir <dir>` | write straight into a scripts/addons directory |
| `-Version 0.4.0` | install a different release |
| `-AssetBaseUrl <url>` | pull the assets from a mirror instead of GitHub |
| `-SkipAssets` | code only (useful if you already have `is3d/native/`) |
| `-Source` | copy the `is3d/` folder from this checkout instead of the release |

## Use

Open the 3D viewport sidebar and pick the **IS3D** tab.

| Control | Meaning |
| --- | --- |
| 参考图 | the reference image |
| 抠除背景 | run the bundled BiRefNet matte first (turn off for images that already have a clean silhouette) |
| 产出 | `高斯点云` (GPUSplat), `高模` (dense mesh with vertex colours), `低模` (UV unwrap + baked base colour/normal) |
| 输出目录 | where the job folder is written |

The job runs in a separate background Blender process; the UI shows progress and
can cancel. The result is appended to the current scene at the 3D cursor.

## How it works

See `docs/ARCHITECTURE.md` for the stage map, the memory strategy and the
kernel inventory. In short:

1. **preprocess** -- resize, optional BiRefNet matte, subject-bbox crop.
2. **encode** -- DINOv3 ViT-H/16+ and the FLUX.2 VAE encoder produce the image
   tokens and the latent grid.
3. **denoise** -- 50-step rectified-flow sampler in fp16 with classifier-free
   guidance.
4. **octree / gaussians** -- anchors are placed on a predicted octree, each
   anchor expands to 32 Gaussians (position, colour, scale, rotation, opacity).
5. **export** -- Gaussian PLY plus a camera JSON, imported by Blender's native
   raw-splat point cloud and then optionally meshed.

Every stage loads only the weights it needs and frees them before the next one
starts, so the footprint stays near 2 GiB instead of the 3.5 GiB of on-disk
weights.

## Build from source

`docs/BUILD.md` describes the MSVC + CUDA 12.6 toolchain, how the kernel PTX is
compiled with NVRTC (no `nvcc` needed), how `idol_weights.bin` is packed from the
upstream checkpoints, and how to produce the release assets.

## Validation

`validation/` contains the checks used during development: layer-by-layer
comparison of the native BiRefNet port against the PyTorch reference
(`check_birefnet.py`, cosine 0.99997 on the alpha channel), an end-to-end engine
run with and without matting (`run_matte.py`), and a Blender integration test
that generates an asset and renders two views (`blender_package_check.py`).

## Licence

Code: MIT (see LICENSE). Model weights: see NOTICE.md.
