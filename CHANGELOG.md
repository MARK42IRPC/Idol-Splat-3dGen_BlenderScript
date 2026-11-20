# Changelog

## 0.4.0 - first public build

Blender add-on
- one reference image -> Gaussian splat, high-poly or low-poly mesh, appended to
  the current scene at the 3D cursor
- job runs in a separate headless Blender, the UI stays responsive and can cancel
- high-poly path: voxel reconstruction plus Gaussian colour transfer
- low-poly path: automatic UV unwrap with baked base colour and normal maps
- new "抠除背景" toggle drives the bundled BiRefNet matte

Engine
- native CUDA-driver engine (no torch, no ONNX runtime, no CUDA toolkit); the
  kernels are embedded as PTX and JIT-compiled at load time
- complete BiRefNet Swin-L + deformable-ASPP port for background removal
  (alpha cosine 0.99997 against the PyTorch reference, IoU 0.9992)
- staged weight loading: each stage frees its tensors before the next one starts
- measured peak VRAM 1.99 GiB with matting, 1.59 GiB without
