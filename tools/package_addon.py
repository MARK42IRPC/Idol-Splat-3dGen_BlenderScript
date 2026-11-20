"""Build dist/is3d-<version>.zip with the engine and weights bundled in."""

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "is3d"
BUILD = ROOT / "native" / "engine" / "build"
DLL = BUILD / "idol_engine.dll"
WEIGHTS = BUILD / "idol_weights.bin"
SKIP_DIRS = {"__pycache__", "native"}


def version():
    text = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'"version":\s*\((\d+),\s*(\d+),\s*(\d+)\)', text)
    if not match:
        raise SystemExit("bl_info version not found")
    return ".".join(match.groups())


def main():
    missing = [str(path) for path in (DLL, WEIGHTS) if not path.is_file()]
    if missing:
        raise SystemExit("missing build output: " + ", ".join(missing))
    out = ROOT / "dist" / f"is3d-{version()}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted(
        path for path in PACKAGE.rglob("*")
        if path.is_file() and not SKIP_DIRS.intersection(path.relative_to(PACKAGE).parts)
    )
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sources:
            archive.write(path, f"is3d/{path.relative_to(PACKAGE).as_posix()}")
        archive.write(DLL, "is3d/native/idol_engine.dll", compress_type=zipfile.ZIP_STORED)
        archive.write(WEIGHTS, "is3d/native/idol_weights.bin",
                      compress_type=zipfile.ZIP_STORED)
    print(f"{out}  {out.stat().st_size / 2**30:.2f} GiB  ({len(sources)} python files)")


if __name__ == "__main__":
    main()
