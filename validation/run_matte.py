"""End-to-end check of the BiRefNet matte path inside the shipped engine."""
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_engine", ROOT / "is3d" / "native_engine.py")
ne = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ne)

GOLD = ROOT / "validation" / "results" / "birefnet-golden"


def load_rgba(path):
    image = Image.open(path).convert("RGBA")
    rgba = np.ascontiguousarray(np.asarray(image, np.float32) / 255.0)
    return rgba


def prep_alpha(engine, rgba):
    lib = engine._lib
    lib.idol_preprocess_mode.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int, C.c_int, C.c_float,
                                         C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.idol_preprocess_mode.restype = C.c_int32
    h, w = rgba.shape[:2]
    prepared = np.zeros(3 * 1024 * 1024, np.float32)
    alpha = np.zeros(1024 * 1024, np.float32)
    rc = lib.idol_preprocess_mode(engine._handle, rgba.ctypes.data_as(C.POINTER(C.c_float)), w, h, 1.2,
                                  prepared.ctypes.data_as(C.POINTER(C.c_float)),
                                  alpha.ctypes.data_as(C.POINTER(C.c_float)))
    if rc:
        raise RuntimeError(engine._error())
    return prepared, alpha[:h * w].reshape(h, w)


def main():
    rgba = load_rgba(ROOT / "validation" / "fixtures" / "fig-snowman.webp")
    ref = np.load(GOLD / "snowman.npz")["alpha"][0, 0]
    report = {}
    for flag in (True, False):
        with ne.Engine() as engine:
            engine.set_crop_margin(1.2)
            engine.set_remove_background(flag)
            engine.reset_peak()
            prepared, alpha = prep_alpha(engine, rgba)
            used, peak = engine.memory()
            key = "matte" if flag else "crop"
            entry = {"peak_mib": round(peak, 1), "coverage": float((alpha > 0.5).mean())}
            if flag:
                # The golden reference ran on the 1024x1024 view of the same picture; compare after
                # matching the frame size the engine worked in.
                smaller = np.asarray(Image.fromarray((ref * 255).astype(np.uint8)).resize(
                    (alpha.shape[1], alpha.shape[0]), Image.BILINEAR), np.float32) / 255.0
                ma, mb = smaller > 0.5, alpha > 0.5
                entry["iou"] = float((ma & mb).sum() / (ma | mb).sum())
                entry["mae"] = float(np.abs(smaller - alpha).mean())
            report[key] = entry
            np.save(ROOT / "validation" / f"_matte_{key}.npy", alpha)
            print(key, entry, flush=True)
            stats = engine.generate(rgba, rgba.shape[1], rgba.shape[0],
                                    ROOT / "validation" / f"_matte_{key}.ply",
                                    ROOT / "validation" / f"_matte_{key}.json",
                                    seed=7, steps=2, points=32768,
                                    progress=lambda percent, message: None)
            report[key].update({"seconds": round(stats["seconds"], 2),
                                "run_peak_mib": round(stats["peak_mib"], 1),
                                "resident_mib": round(stats["resident_mib"], 1)})
            print(key, report[key], flush=True)
    (ROOT / "validation" / "results" / "matte-report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
