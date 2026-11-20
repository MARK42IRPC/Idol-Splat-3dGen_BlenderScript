"""Stage-by-stage checks of the native Idol engine against the reference probe tensors."""
import ctypes as C
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "validation" / "probe"
DLL = ROOT / "native" / "engine" / "build" / "idol_engine.dll"
WEIGHTS = ROOT / "native" / "engine" / "build" / "idol_weights.bin"

F32 = C.c_float
P32 = C.POINTER(C.c_float)


def load():
    lib = C.WinDLL(str(DLL))
    lib.idol_engine_abi.restype = C.c_uint32
    lib.idol_engine_create.argtypes = [C.c_int32, C.c_char_p, C.POINTER(C.c_void_p)]
    lib.idol_engine_create.restype = C.c_int32
    lib.idol_engine_destroy.argtypes = [C.c_void_p]
    lib.idol_engine_info.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.c_int32),
                                     C.POINTER(C.c_int32), C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]
    lib.idol_engine_info.restype = C.c_int32
    lib.idol_memory.argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]
    lib.idol_memory.restype = C.c_int32
    lib.idol_engine_unload.argtypes = [C.c_void_p]
    lib.idol_engine_unload.restype = C.c_int32
    lib.idol_debug_gemm.argtypes = [C.c_void_p, P32, C.c_int, C.c_int, P32, C.c_int, P32]
    lib.idol_debug_gemm.restype = C.c_int32
    lib.idol_debug_flash.argtypes = [C.c_void_p, P32, P32, P32, C.c_int, C.c_int, P32]
    lib.idol_debug_flash.restype = C.c_int32
    lib.idol_encode_dino.argtypes = [C.c_void_p, P32, P32]
    lib.idol_encode_dino.restype = C.c_int32
    lib.idol_encode_vae.argtypes = [C.c_void_p, P32, P32]
    lib.idol_encode_vae.restype = C.c_int32
    lib.idol_encode.argtypes = [C.c_void_p, P32, P32, P32]
    lib.idol_encode.restype = C.c_int32
    lib.idol_velocity.argtypes = [C.c_void_p, P32, P32, P32, P32, F32, P32, P32]
    lib.idol_velocity.restype = C.c_int32
    lib.idol_octree.argtypes = [C.c_void_p, P32, C.c_int, C.c_int, C.c_uint64, P32]
    lib.idol_octree.restype = C.c_int32
    lib.idol_gaussians.argtypes = [C.c_void_p, P32, P32, C.c_int, P32, P32, P32, P32, P32]
    lib.idol_gaussians.restype = C.c_int32
    lib.idol_engine_last_error.restype = C.c_char_p
    return lib


def f32(array):
    return np.ascontiguousarray(array, dtype=np.float32)


def err(lib, tag):
    print(f"{tag}: FAILED -> {lib.idol_engine_last_error().decode('utf-8', 'replace')}")


def cosine(a, b):
    a = a.reshape(a.shape[0], -1).astype(np.float64)
    b = b.reshape(b.shape[0], -1).astype(np.float64)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


def check_gemm(lib, eng):
    rng = np.random.default_rng(0)
    m, k, n = 64, 48, 80
    a = rng.standard_normal((m, k)).astype(np.float32) * 0.5
    b = rng.standard_normal((k, n)).astype(np.float32) * 0.5
    c = np.zeros((m, n), dtype=np.float32)
    rc = lib.idol_debug_gemm(eng, a.ctypes.data_as(P32), m, k, b.ctypes.data_as(P32), n,
                             c.ctypes.data_as(P32))
    if rc:
        return err(lib, "gemm")
    ref = a.astype(np.float16).astype(np.float32) @ b.astype(np.float16).astype(np.float32)
    print(f"gemm   : max abs diff {np.abs(c - ref).max():.4f}  (rel {np.linalg.norm(c - ref)/np.linalg.norm(ref):.2e})")


def check_flash(lib, eng):
    rng = np.random.default_rng(1)
    l, h = 128, 2
    q = (rng.standard_normal((l, h, 64)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((l, h, 64)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((l, h, 64)) * 0.5).astype(np.float32)
    out = np.zeros((l, h, 64), dtype=np.float32)
    rc = lib.idol_debug_flash(eng, q.ctypes.data_as(P32), k.ctypes.data_as(P32), v.ctypes.data_as(P32),
                              l, h, out.ctypes.data_as(P32))
    if rc:
        return err(lib, "flash")
    s = np.einsum("lhd,khd->lhk", q, k) / np.sqrt(64.0)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    ref = np.einsum("lhk,khd->lhd", p, v)
    print(f"flash  : max abs diff {np.abs(out - ref).max():.5f}  (rel {np.linalg.norm(out-ref)/np.linalg.norm(ref):.2e})")


def check_encode(lib, eng):
    from PIL import Image
    img = np.asarray(Image.open(PROBE / "prepared.png").convert("RGB"), dtype=np.float32) / 255.0
    prepared = np.ascontiguousarray(img.transpose(2, 0, 1))
    f1 = np.zeros((4101, 1280), dtype=np.float32)
    f2 = np.zeros((4101, 128), dtype=np.float32)
    import time
    t0 = time.time()
    rc = lib.idol_encode_dino(eng, prepared.ctypes.data_as(P32), f1.ctypes.data_as(P32))
    print(f"  dino   : {time.time() - t0:.1f}s")
    if rc:
        return err(lib, "encode-dino")
    ref1 = np.load(PROBE / "feature1.npy").reshape(4101, 1280)
    print(f"  feature1 cosine {cosine(f1, ref1):.6f}  rel {np.linalg.norm(f1-ref1)/np.linalg.norm(ref1):.4f}")
    np.save(PROBE / "native_feature1.npy", f1)
    rc = lib.idol_encode_vae(eng, prepared.ctypes.data_as(P32), f2.ctypes.data_as(P32))
    print(f"  vae    : {time.time() - t0:.1f}s")
    if rc:
        return err(lib, "encode-vae")
    ref1 = np.load(PROBE / "feature1.npy").reshape(4101, 1280)
    ref2 = np.load(PROBE / "feature2.npy").reshape(4101, 128)
    print(f"  feature1 cosine {cosine(f1, ref1):.6f}  rel {np.linalg.norm(f1-ref1)/np.linalg.norm(ref1):.4f}")
    print(f"  feature2 cosine {cosine(f2[5:], ref2[5:]):.6f}  rel {np.linalg.norm(f2[5:]-ref2[5:])/np.linalg.norm(ref2[5:]):.4f}")
    np.save(PROBE / "native_feature1.npy", f1)
    np.save(PROBE / "native_feature2.npy", f2)


def check_velocity(lib, eng):
    f1 = np.load(PROBE / "feature1.npy").reshape(4101, 1280)
    f2 = np.load(PROBE / "feature2.npy").reshape(4101, 128)
    lat = np.load(PROBE / "noise_latent.npy").reshape(8192, 16)
    cam = np.load(PROBE / "noise_camera.npy").reshape(5)
    out = np.zeros((8192, 16), dtype=np.float32)
    ocam = np.zeros(5, dtype=np.float32)
    import time
    t0 = time.time()
    rc = lib.idol_velocity(eng, f1.ctypes.data_as(P32), f2.ctypes.data_as(P32), lat.ctypes.data_as(P32),
                           cam.ctypes.data_as(P32), 1000.0, out.ctypes.data_as(P32), ocam.ctypes.data_as(P32))
    if rc:
        return err(lib, "velocity")
    print(f"velocity: {time.time() - t0:.1f}s")
    ref = np.load(PROBE / "velocity_latent.npy").reshape(8192, 16)
    refcam = np.load(PROBE / "velocity_camera.npy").reshape(5)
    print(f"  latent cosine {cosine(out, ref):.6f}  rel {np.linalg.norm(out-ref)/np.linalg.norm(ref):.4f}  max {np.abs(out-ref).max():.4f}")
    print(f"  ref[0,:6] {refcam}")
    print(f"  got[0,:6] {ocam}")
    np.save(PROBE / "native_velocity.npy", out)


def main():
    steps = sys.argv[1:] or ["gemm", "flash", "encode", "velocity"]
    lib = load()
    eng = C.c_void_p()
    rc = lib.idol_engine_create(0, str(WEIGHTS).encode("utf-8"), C.byref(eng))
    if rc:
        print("create failed:", lib.idol_engine_last_error().decode("utf-8", "replace"))
        return 1
    name = C.create_string_buffer(256)
    major, minor = C.c_int32(), C.c_int32()
    freeb, totalb = C.c_uint64(), C.c_uint64()
    lib.idol_engine_info(eng, name, 256, C.byref(major), C.byref(minor), C.byref(freeb), C.byref(totalb))
    print(f"device : {name.value.decode()}  sm_{major.value}{minor.value}  free {freeb.value/2**30:.2f} GiB")
    for step in steps:
        fn = globals().get(f"check_{step}")
        if fn is None:
            print("unknown step", step)
            continue
        fn(lib, eng)
        used, peak = C.c_uint64(), C.c_uint64()
        lib.idol_memory(eng, C.byref(used), C.byref(peak))
        print(f"  vram  : resident {used.value/2**20:.0f} MiB  peak {peak.value/2**20:.0f} MiB")
    lib.idol_engine_destroy(eng)
    return 0


if __name__ == "__main__":
    sys.exit(main())
