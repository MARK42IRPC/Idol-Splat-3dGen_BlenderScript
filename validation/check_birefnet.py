'''Compare the native BiRefNet forward against the torch golden dump, layer by layer.'''
import ctypes as C
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "native" / "engine" / "build"
GOLDEN = ROOT / "validation" / "results" / "birefnet-golden" / "snowman.npz"
DUMP = ROOT / "validation" / "results" / "birefnet-golden" / "native"

P32 = C.POINTER(C.c_float)


def engine():
    lib = C.WinDLL(str(BUILD / "idol_engine.dll"))
    lib.idol_engine_create.argtypes = [C.c_int32, C.c_char_p, C.POINTER(C.c_void_p)]
    lib.idol_engine_create.restype = C.c_int32
    lib.idol_engine_last_error.restype = C.c_char_p
    lib.idol_engine_set_crop_margin.argtypes = [C.c_void_p, C.c_float]
    lib.idol_engine_set_remove_background.argtypes = [C.c_void_p, C.c_int32]
    lib.idol_memory.argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]
    lib.idol_preprocess_mode.argtypes = [C.c_void_p, P32, C.c_int, C.c_int, C.c_float, P32, P32]
    lib.idol_preprocess_mode.restype = C.c_int32
    handle = C.c_void_p()
    if lib.idol_engine_create(0, str(BUILD / "idol_weights.bin").encode(), C.byref(handle)):
        raise SystemExit(lib.idol_engine_last_error().decode())
    return lib, handle


def main():
    golden = np.load(GOLDEN)
    inv_std = np.array([0.229, 0.224, 0.225], np.float32)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    rgb = golden["input"][0].astype(np.float32) * inv_std[:, None, None] + mean[:, None, None]
    rgba = np.zeros((1024, 1024, 4), np.float32)
    rgba[:, :, :3] = np.clip(rgb.transpose(1, 2, 0), 0.0, 1.0)
    rgba[:, :, 3] = 1.0

    DUMP.mkdir(parents=True, exist_ok=True)
    for f in DUMP.glob("*.f32"):
        f.unlink()
    os.environ["IDOL_BFNET_DUMP"] = str(DUMP)

    lib, handle = engine()
    lib.idol_engine_set_crop_margin(handle, C.c_float(1.2))
    lib.idol_engine_set_remove_background(handle, 1)
    lib.idol_engine_reset_peak(handle) if hasattr(lib, "idol_engine_reset_peak") else None
    prepared = np.zeros(3 * 1024 * 1024, np.float32)
    alpha = np.zeros(1024 * 1024, np.float32)
    rc = lib.idol_preprocess_mode(handle, rgba.ctypes.data_as(P32), 1024, 1024, C.c_float(1.2),
                                  prepared.ctypes.data_as(P32), alpha.ctypes.data_as(P32))
    if rc:
        raise SystemExit(lib.idol_engine_last_error().decode())
    used, peak = C.c_uint64(), C.c_uint64()
    lib.idol_memory(handle, C.byref(used), C.byref(peak))
    print(f"peak {peak.value / 2**20:.0f} MiB")

    rows = []
    for key in golden.files:
        if key == "input":
            continue
        gold = golden[key].astype(np.float32)[0]
        ch, h, w = gold.shape
        path = DUMP / f"{key}.f32"
        if not path.is_file():
            rows.append((key, float("nan"), float("nan")))
            continue
        flat = np.fromfile(path, np.float32)
        got = flat.reshape(h * w, -1)[:, :ch].T.reshape(ch, h, w)
        a = got.ravel()
        b = gold.ravel()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos = float(a @ b / denom) if denom else 1.0
        scale = max(1e-6, float(np.abs(b).max()))
        rel = float(np.abs(a - b).max() / scale)
        rows.append((key, cos, rel))
    print(f"{'layer':10s} {'cosine':>10s} {'maxrel':>9s}")
    worst = 1.0
    for key, cos, rel in rows:
        print(f"{key:10s} {cos:10.6f} {rel:9.4f}")
        if cos == cos:
            worst = min(worst, cos)
    print(f"worst cosine {worst:.6f}")


if __name__ == "__main__":
    main()
