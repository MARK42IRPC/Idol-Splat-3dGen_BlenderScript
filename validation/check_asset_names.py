"""Check the optional asset-name field: folder naming, numbering and library titles."""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from is3d import library  # noqa: E402 - needs bpy


def _finish_job(folder, name=""):
    (folder / "job.json").write_text(json.dumps({"output_kind": "SPLAT", "points": 32768,
                                                 "name": name}), encoding="utf-8")
    (folder / library.TRANSFER_FILE).write_bytes(b"stub")


def main():
    root = Path(tempfile.mkdtemp(prefix="is3d-names-"))
    checks = []

    def expect(label, value, wanted):
        checks.append((label, value, wanted))
        print(f"{'ok  ' if value == wanted else 'FAIL'} {label}: {value!r}", flush=True)

    folder, title = library.new_job_folder(root, "")
    expect("unnamed run is numbered", folder.name, "is3d-0001")
    expect("title follows the folder", title, folder.name)
    _finish_job(folder, title)

    folder, title = library.new_job_folder(root, "")
    expect("second unnamed run increments", folder.name, "is3d-0002")
    _finish_job(folder, title)

    folder, title = library.new_job_folder(root, "  雪人 / 测试  ")
    expect("named run is sanitised", folder.name, "雪人 _ 测试")
    _finish_job(folder, title)

    folder, _ = library.new_job_folder(root, "雪人 / 测试")
    expect("duplicate name gets a suffix", folder.name, "雪人 _ 测试-2")
    _finish_job(folder, "雪人 _ 测试-2")

    folder, _ = library.new_job_folder(root, "CON")
    expect("reserved name falls back to numbering", folder.name, "is3d-0005")
    _finish_job(folder, folder.name)

    folder, _ = library.new_job_folder(root, ". . .")
    expect("empty after cleanup falls back", folder.name, "is3d-0006")
    _finish_job(folder, folder.name)

    expect("safe name keeps unicode", library.safe_folder_name("偶像-Splat 3D"), "偶像-Splat 3D")
    expect("safe name is length capped", len(library.safe_folder_name("x" * 80)), 48)

    info = library._folder_info(root / "雪人 _ 测试", None)
    expect("library title uses job name", info.title if info else None, "雪人 _ 测试")
    expect("numbered asset keeps its number",
           (library._folder_info(root / "is3d-0001", None) or None).title, "is3d-0001")

    failed = [item for item in checks if item[1] != item[2]]
    print(f"NAMES: {'PASSED' if not failed else 'FAILED'} ({len(checks) - len(failed)}/{len(checks)})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
