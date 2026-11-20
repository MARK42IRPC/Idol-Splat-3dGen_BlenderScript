# Building IS3D from source

## Toolchain

| Tool | Version used |
| --- | --- |
| Visual Studio 2022 (Desktop C++ workload) | 17.x |
| CUDA Toolkit | 12.6 (only for `cuda.lib` and NVRTC, no `nvcc` call) |
| Python | 3.11+ (the venv in `.reference/` was used during development) |

The engine itself never links against the CUDA runtime: it loads `nvcuda.dll`
from the driver at run time and JIT-compiles `kernels.ptx`.

## Engine

```powershell
powershell -ExecutionPolicy Bypass -File native/engine/build_all.ps1
```

Two steps:

1. `build_ptx.py` compiles `kernels.cu` with NVRTC (`compute_86`, `--std=c++17`)
   and `embed_ptx.py` turns the result into `build/ptx_embed.inc`.
2. `engine.cpp` is compiled with MSVC into `build/idol_engine.dll`.

Rebuild only the host side (after editing `engine.cpp` / `birefnet.inc`):

```powershell
powershell -ExecutionPolicy Bypass -File native/engine/build_all.ps1 -SkipPtx
```

## Weights

`native/engine/pack_weights.py` walks the upstream checkpoints listed in
NOTICE.md, keeps only the tensors the inference path touches, transposes the
conv/linear weights into the layout the kernels expect, and writes one
`build/idol_weights.bin` (currently 3.54 GB, 2349 tensors). Paths are configured
at the top of the script.

Unused heads (for example `bb.patch_embed.mask_token`, every
`relative_position_index`) are dropped, which keeps the download smaller and the
resident set predictable.

## Release assets

GitHub rejects single assets above 2 GiB, so the weight file is published in
parts:

```powershell
python tools/make_release_assets.py            # -> dist/release/*
```

That writes `idol_engine.dll`, `idol_weights.bin.partNN`, `SHA256SUMS.txt` and
`manifest.json`. Upload them to a release:

```powershell
gh release create v0.4.0 dist/release/* --title "IS3D 0.4.0" --notes "First public build"
```

`install/install.ps1` downloads that set, concatenates the parts and checks every
SHA-256 against `SHA256SUMS.txt`.

## Add-on zip

`tools/package_addon.py` produces a single self-contained
`dist/is3d-<version>.zip` (add-on code + DLL + weights) for users who would
rather not run the installer.
