# Architecture

## Process model

```
Blender UI (is3d/__init__.py, async_ui.py)
      |  writes job.json, spawns
      v
Blender -b (is3d/worker.py)  --bpy.data.libraries.write--> asset_transfer.blend
      |
      v
idol_engine.dll (native/engine/*)
```

The UI thread never blocks: `async_ui.start_generation` writes a job description
and starts a second headless Blender, then polls for progress lines. The worker
runs the engine, imports the result with `bpy.ops.wm.ply_import`, optionally
reconstructs a mesh, and writes the finished object into a transfer `.blend`
that the UI appends to the current scene at the 3D cursor.

## Engine stages

| Stage | Kernel work | Resident weights |
| --- | --- | --- |
| `preprocess` | bilinear resize, BiRefNet matte (Swin-L + deformable ASPP), bbox crop | `birefnet.*` |
| `encode` | DINOv3 ViT-H/16+ (24 blocks), FLUX.2 VAE encoder | `dinov3.*`, `vae.*` |
| `denoise` | 50-step flow sampler, DiT with per-block attention | `denoiser.*` |
| `octree` | latent -> anchor positions | `denoiser.*` |
| `gaussians` | anchor -> 32 Gaussians each | `decoder_gs.*` |

`free_weights_prefix()` drops a stage's tensors as soon as the next stage starts,
and the device allocator is a size-indexed pool with `trim()` between stages.
Measured peak: 1.99 GiB with matte, 1.59 GiB without.

## Kernels

`native/engine/kernels.cu` is compiled by NVRTC into PTX and JIT-ed at load.

- `k_gemm2` / `k_gemm_nt` -- fp16 tensor-core GEMM used for every projection and
  convolution (convolutions are im2col + GEMM).
- `k_flash3` -- streaming attention (64 queries per block, 64 keys per tile,
  fp32 softmax state), used by the DiT, DINOv3 and the Swin backbone.
- `k_swin_attn` -- 12x12 window attention with head_dim 32 for BiRefNet, K/V kept
  in shared memory, results written straight back into token order.
- `k_deform_im2col_s` -- deformable gather for the ASPP branches; the offset pair
  layout matches torchvision (`dy, dx` interleaved per tap).
- `k_im2col_s`, `k_patch_merge_concat`, `k_resize`, `k_gate_mul`, ... -- the
  remaining plumbing.

## Numerical validation

`validation/birefnet_ref.py` dumps every intermediate of the PyTorch BiRefNet
reference for one input, `validation/check_birefnet.py` runs the same input
through the DLL and compares layer by layer (`IDOL_BFNET_DUMP` enables the
native dumps). Current agreement: encoder blocks 0.975-0.998 cosine, decoder
0.986-0.99, alpha 0.99997 (IoU 0.9992).
