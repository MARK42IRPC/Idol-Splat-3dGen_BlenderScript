"""Run one named job end to end (add-on path) and inspect folder, object and library title."""

import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bpy  # noqa: E402

is3d = importlib.import_module("is3d")
is3d.register()
async_ui = importlib.import_module("is3d.async_ui")
library = importlib.import_module("is3d.library")

OUTPUT = Path("C:/is3d_ui_test/assets")
NAME = "雪人命名"


def main():
    scene = bpy.context.scene
    options = scene.is3d_mesh_options
    options.reference_image = str(ROOT / "validation" / "fixtures" / "fig-snowman.webp")
    options.output_kind = "SPLAT"
    options.output_directory = str(OUTPUT)
    options.remove_background = True
    options.steps = 8
    options.gaussian_count = "32768"
    options.asset_name = NAME

    folder = async_ui.start_generation(scene)
    print("FOLDER", folder.name, flush=True)
    deadline = time.time() + 900
    while async_ui.is_busy() and time.time() < deadline:
        async_ui.poll_generation()
        time.sleep(0.25)
    print("STATE", options.job_state, options.job_message, flush=True)
    if options.job_state != "FINISHED":
        raise SystemExit("job did not finish")

    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    objects = [obj for obj in scene.objects if obj.name == NAME]
    library.invalidate()
    titles = [item.title for item in library.scan(scene) if item.folder == str(folder)]
    checks = [
        ("folder name", folder.name, NAME),
        ("job.json name", job.get("name"), NAME),
        ("object in scene", objects[0].name if objects else None, NAME),
        ("library title", titles[0] if titles else None, NAME),
    ]
    failed = [line for line in checks if line[1] != line[2]]
    for label, value, wanted in checks:
        print(f"{'ok  ' if value == wanted else 'FAIL'} {label}: {value!r}", flush=True)
    print(f"NAMED: {'PASSED' if not failed else 'FAILED'}", flush=True)
    return 1 if failed else 0


try:
    code = main()
finally:
    is3d.unregister()
raise SystemExit(code)
