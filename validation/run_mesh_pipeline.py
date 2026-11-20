"""Run the IS3D mesh workflow against a saved Gaussian Splat scene in Blender."""

import argparse
import json
from pathlib import Path
import sys

import bpy


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from is3d.pipeline import HIGH_METHODS, LOW_METHODS, is_gaussian, run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", help="Name of the Gaussian Splat object if the scene has several")
    parser.add_argument("--high", choices=[x[0] for x in HIGH_METHODS], default="AUTO")
    parser.add_argument("--low", choices=[x[0] for x in LOW_METHODS], default="AUTO")
    parser.add_argument("--voxel-size", type=float, default=0.006)
    parser.add_argument("--triangles", type=int, default=20000)
    parser.add_argument("--resolution", type=int, default=2048)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    candidates = [obj for obj in bpy.context.scene.objects if is_gaussian(obj)]
    if args.source:
        candidates = [obj for obj in candidates if obj.name == args.source]
    if len(candidates) != 1:
        raise ValueError("Open a scene with one Gaussian Splat or specify --source OBJECT_NAME")

    output = args.output.resolve()
    path = output / "asset_mesh.blend"
    if path.exists():
        raise FileExistsError(path)
    _, report = run_pipeline(candidates[0], output, high_method=args.high, low_method=args.low,
                             voxel_size=args.voxel_size, target_triangles=args.triangles,
                             resolution=args.resolution, progress=lambda message: print(message, flush=True))
    if bpy.ops.wm.save_as_mainfile(filepath=str(path), copy=True) != {"FINISHED"}:
        raise RuntimeError(f"Blender failed to save {path}")
    print(json.dumps({"blend": str(path), "report": report}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
