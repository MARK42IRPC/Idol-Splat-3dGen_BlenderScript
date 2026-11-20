---
license: mit
language:
  - en
  - zh
tags:
  - blender
  - addon
  - gaussian-splatting
  - 3d-generation
  - single-image
library_name: is3d
---

# IS3D - Idol Splat 3D Generator weights

**English** | [简体中文](#中文)

Model assets for the [IS3D Blender add-on](https://github.com/MARK42IRPC/is3D-BlenderScript):
one reference image in, Gaussian splat / high-poly / low-poly out, on the GPU
through the NVIDIA driver only -- no PyTorch, no ONNX Runtime, no CUDA Toolkit.

This repository is a **download mirror**, not a library. The files are byte
identical to the GitHub release, so `SHA256SUMS.txt` covers all mirrors.

## Files

| File | Size | Purpose |
| --- | --- | --- |
| `idol_engine.dll` | 1.2 MB | native inference engine (host code + embedded PTX `compute_86`) |
| `idol_weights.bin.part00/01` | 1.7 GiB + 1.6 GiB | weight container `IDOLW1`, 2349 tensors, split because GitHub caps assets at 2 GiB |
| `is3d-addon-0.4.0.zip` | 40 KB | the Blender add-on code (no weights) |
| `SHA256SUMS.txt` | 437 B | checksums for every file above |
| `manifest.json` | small | sizes, part list, checksum map |

The installer concatenates `part00` + `part01` into `idol_weights.bin`
(3,537,766,912 bytes) and verifies the SHA-256 of both the parts and the result.

## Install

```powershell
git clone https://github.com/MARK42IRPC/is3D-BlenderScript
cd is3D-BlenderScript
powershell -ExecutionPolicy Bypass -File install/install.ps1 -Mirror hf
```

Requirements: Windows 10/11 x64, NVIDIA RTX 30/40/50, Blender 5.3+, ~4 GB VRAM,
~4 GB disk.

## Licence

Add-on code is MIT. The weights are a repack of upstream research models --
idol (speridlabs), FLUX.2 klein VAE (Black Forest Labs), DINOv3 (Meta) and
BiRefNet -- and carry their licences; see `NOTICE.md` in the GitHub repository.
Research and personal use only.

---

## 中文

**IS3D（Idol Splat 3D Generator）Blender 插件的模型资源。**

一张参考图进，高斯点云 / 高模 / 低模出，全程只用 NVIDIA 驱动推理：不需要
PyTorch、不需要 ONNX Runtime、不需要 CUDA Toolkit。

本仓库是**下载镜像**，文件与 GitHub Release 完全一致，`SHA256SUMS.txt` 通用。

安装：

```powershell
powershell -ExecutionPolicy Bypass -File install/install.ps1 -Mirror hf
```

也可以直接把 `-AssetBaseUrl` 指向本仓库：

```
https://hf-mirror.com/Mark42IRP/is3D-BlenderScript/resolve/main
```

代码 MIT；权重是对上游研究模型（idol、FLUX.2 klein VAE、DINOv3、BiRefNet）的
重新打包，遵循各自许可，仅供研究与个人使用。
