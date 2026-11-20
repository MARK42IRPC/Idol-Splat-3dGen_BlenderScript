# Changelog

## 0.4.1 - panel and asset library

- reference image card with a live thumbnail, resolution and file size
- collapsible asset library: every finished job is listed with the preview that
  the worker renders (384 px, framed on the object only), so a previous result
  can be reused without running the model again
- drag a library entry from the hand button straight into the 3D view, or click
  add to drop it on the 3D cursor; reveal and delete are one click away
- generation runs still block the inputs, but the panel now shows a progress bar
  in the same place the results appear

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
