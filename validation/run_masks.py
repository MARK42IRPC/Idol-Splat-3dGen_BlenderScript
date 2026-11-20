"""Extract the bundled BiRefNet matte for a set of pictures and save binary masks.

Usage:
    python validation/run_masks.py [--out DIR] image1.png image2.jpg ...

For every picture the engine's own preprocess pass produces the soft alpha the generator
would use; this script saves the soft alpha, the binary mask (alpha > 0.5) and a four panel
comparison sheet, plus a JSON summary with coverage, soft-edge share and bounding box.
"""

import argparse
import ctypes as C
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402 - loaded by path so the add-on package (which needs bpy) is skipped

_spec = importlib.util.spec_from_file_location("native_engine", ROOT / "is3d" / "native_engine.py")
ne = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ne)

F32P = C.POINTER(C.c_float)


def bind(engine):
    lib = engine._lib
    lib.idol_preprocess_mode.argtypes = [C.c_void_p, F32P, C.c_int, C.c_int, C.c_float, F32P, F32P]
    lib.idol_preprocess_mode.restype = C.c_int32
    return lib


def load_flat(path, background=None):
    """RGBA float32 of the picture; a stored alpha is composited away so BiRefNet must work."""
    image = Image.open(path).convert("RGBA")
    rgba = np.asarray(image, np.float32) / 255.0
    if background is not None and float(rgba[..., 3].min()) < 0.999:
        alpha = rgba[..., 3:4]
        rgba = np.ascontiguousarray(rgba[..., :3] * alpha + np.asarray(background, np.float32) * (1.0 - alpha))
        rgba = np.ascontiguousarray(np.concatenate([rgba, np.ones_like(alpha)], axis=2))
    return rgba


def matte(engine, lib, rgba, margin=1.2):
    height, width = rgba.shape[:2]
    prepared = np.zeros(3 * 1024 * 1024, np.float32)
    alpha = np.zeros(width * height, np.float32)
    rc = lib.idol_preprocess_mode(
        engine._handle, rgba.ctypes.data_as(F32P), width, height, float(margin),
        prepared.ctypes.data_as(F32P), alpha.ctypes.data_as(F32P))
    if rc:
        raise RuntimeError(engine._error())
    return alpha.reshape(height, width)


def stats(alpha, binary):
    coverage = float(binary.mean())
    soft = float(((alpha > 0.02) & (alpha < 0.98)).mean())
    rows, columns = np.nonzero(binary)
    box = [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())] if len(rows) else None
    return {"coverage": round(coverage, 4), "soft_edge": round(soft, 4), "bbox": box}


def checkerboard(width, height, cell=16):
    yy, xx = np.mgrid[0:height, 0:width]
    board = (((yy // cell) + (xx // cell)) % 2).astype(np.float32)
    return (0.32 + 0.10 * board)[..., None].repeat(3, axis=2)


def panel(rgba, alpha, binary, title, height=320):
    rgb = rgba[..., :3]
    cutout = rgb * alpha[..., None] + checkerboard(rgba.shape[1], rgba.shape[0]) * (1.0 - alpha[..., None])
    mask = np.repeat(binary[..., None].astype(np.float32), 3, axis=2)
    tiles = [rgb, np.repeat(alpha[..., None], 3, axis=2), mask, cutout]
    images = [Image.fromarray((np.clip(tile, 0.0, 1.0) * 255).astype(np.uint8)) for tile in tiles]
    scale = height / images[0].height
    images = [image.resize((max(1, int(image.width * scale)), height), Image.LANCZOS) for image in images]
    gap = 6
    sheet = Image.new("RGB", (sum(image.width for image in images) + gap * 3, height + 22), (18, 18, 22))
    x = 0
    for image in images:
        sheet.paste(image, (x, 22))
        x += image.width + gap
    from PIL import ImageDraw

    ImageDraw.Draw(sheet).text((4, 5), title, fill=(232, 232, 236))
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "results" / "mask-tests")
    parser.add_argument("--background", default="128", help="gray level used to flatten stored alpha")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    gray = int(args.background) / 255.0
    report = {}
    sheets = []
    with ne.Engine() as engine:
        lib = bind(engine)
        for path in args.images:
            rgba = load_flat(path, background=(gray, gray, gray))
            engine.reset_peak()
            alpha = matte(engine, lib, rgba)
            _, peak = engine.memory()
            binary = alpha > 0.5
            stem = path.stem
            np.save(out / f"{stem}-alpha.npy", alpha.astype(np.float32))
            Image.fromarray((binary * 255).astype(np.uint8)).save(out / f"{stem}-mask.png")
            Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8)).save(out / f"{stem}-alpha.png")
            sheet = panel(rgba, alpha, binary, f"{stem}  peak {peak:.0f} MiB")
            sheet.save(out / f"{stem}-panel.png")
            sheets.append(sheet)
            report[stem] = {"size": [rgba.shape[1], rgba.shape[0]], "peak_mib": round(peak, 1),
                            "seconds": None, **stats(alpha, binary)}
            print(stem, report[stem], flush=True)
    if sheets:
        width = max(sheet.width for sheet in sheets)
        sheet_all = Image.new("RGB", (width, sum(sheet.height for sheet in sheets)), (18, 18, 22))
        y = 0
        for sheet in sheets:
            sheet_all.paste(sheet, (0, y))
            y += sheet.height
        sheet_all.save(out / "masks-contact-sheet.png")
    (out / "masks-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
