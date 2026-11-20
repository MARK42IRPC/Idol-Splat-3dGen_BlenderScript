"""Prepare the GitHub release payload: engine DLL, split weight file, checksums.

GitHub rejects single release assets above 2 GiB, so ``idol_weights.bin`` is cut
into ``partNN`` slices that ``install/install.ps1`` downloads, verifies and
concatenates back together.

    python tools/make_release_assets.py
    python tools/make_release_assets.py --chunk-gib 1.5
"""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "is3d"
BUILD = ROOT / "native" / "engine" / "build"
DLL = BUILD / "idol_engine.dll"
WEIGHTS = BUILD / "idol_weights.bin"


def version():
    text = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'"version":\s*\((\d+),\s*(\d+),\s*(\d+)\)', text)
    if not match:
        raise SystemExit("bl_info version not found in is3d/__init__.py")
    return ".".join(match.groups())


def digest(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def split(path, out_dir, chunk_bytes):
    parts = []
    with open(path, "rb") as handle:
        index = 0
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            part = out_dir / f"{path.name}.part{index:02d}"
            part.write_bytes(block)
            parts.append(part)
            index += 1
    return parts


def addon_zip(out_dir, ver):
    target = out_dir / f"is3d-addon-{ver}.zip"
    sources = sorted(p for p in PACKAGE.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix == ".py")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sources:
            archive.write(path, f"is3d/{path.relative_to(PACKAGE).as_posix()}")
    return target, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-gib", type=float, default=1.7,
                        help="size of each weight slice (GitHub caps assets at 2 GiB)")
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "release")
    args = parser.parse_args()

    missing = [str(p) for p in (DLL, WEIGHTS) if not p.is_file()]
    if missing:
        raise SystemExit("missing build output: " + ", ".join(missing))

    ver = version()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("idol_weights.bin.part*"):
        stale.unlink()

    dll_copy = out / "idol_engine.dll"
    dll_copy.write_bytes(DLL.read_bytes())

    parts = split(WEIGHTS, out, int(args.chunk_gib * 2 ** 30))
    zip_path, sources = addon_zip(out, ver)

    sums = {}
    for path in [dll_copy, zip_path, *parts]:
        print(f"  hashing {path.name}")
        sums[path.name] = digest(path)
    print("  hashing idol_weights.bin")
    sums["idol_weights.bin"] = digest(WEIGHTS)

    (out / "SHA256SUMS.txt").write_text(
        "".join(f"{sums[name]}  {name}\n" for name in sorted(sums)), encoding="utf-8")

    manifest = {
        "version": ver,
        "engine_dll_bytes": dll_copy.stat().st_size,
        "weights_bytes": WEIGHTS.stat().st_size,
        "weights_parts": [p.name for p in parts],
        "chunk_gib": args.chunk_gib,
        "addon_files": len(sources),
        "sha256": sums,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"\n{out}  {total / 2 ** 30:.2f} GiB  ({len(parts)} weight parts)")
    print("\nUpload with:")
    print(f'  gh release create v{ver} "{out}/*" --title "IS3D {ver}" --notes-file CHANGELOG.md')


if __name__ == "__main__":
    main()
