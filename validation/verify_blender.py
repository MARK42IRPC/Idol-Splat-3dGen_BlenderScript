"""Validate a native Blender 5.3 Gaussian splat imported from an Idol PLY."""

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def ply_vertex_count(path):
    """Read the vertex count independently of Blender's PLY importer."""
    with path.open("rb") as ply:
        if ply.readline(4096).strip() != b"ply":
            raise ValueError(f"Invalid PLY signature in {path}")
        count = None
        for _ in range(1024):
            fields = ply.readline(4096).split()
            if fields == [b"end_header"]:
                if count is None or count < 1:
                    raise ValueError(f"Missing or invalid vertex count in {path}")
                return count
            if len(fields) == 3 and fields[:2] == [b"element", b"vertex"]:
                count = int(fields[2])
    raise ValueError(f"Incomplete PLY header in {path}")


def validate_attributes(cloud, count):
    specs = {
        "position": ("FLOAT_VECTOR", "vector"),
        "scale": ("FLOAT_VECTOR", "vector"),
        "rotation": ("QUATERNION", "value"),
        "radiance:base": ("FLOAT4", "vector"),
    }
    sample_indices = {0, count // 2, count - 1}
    for name, (data_type, value_field) in specs.items():
        attribute = cloud.attributes.get(name)
        if attribute is None or (attribute.domain, attribute.data_type) != ("POINT", data_type):
            raise RuntimeError(f"Expected point-domain {data_type} attribute {name!r}")
        if len(attribute.data) != count:
            raise RuntimeError(f"Attribute {name!r} has {len(attribute.data)} values, expected {count}")
        for index in sample_indices:
            value = getattr(attribute.data[index], value_field)
            if not all(math.isfinite(x) for x in value):
                raise RuntimeError(f"Nonfinite {name!r} at point {index}")
            if name == "scale" and min(value) <= 0.0:
                raise RuntimeError(f"Nonpositive Gaussian scale at point {index}")
            if name == "rotation" and not math.isclose(sum(x * x for x in value), 1.0, abs_tol=1e-4):
                raise RuntimeError(f"Nonunit Gaussian orientation at point {index}")
            if name == "radiance:base" and not 0.0 <= value[3] <= 1.0:
                raise RuntimeError(f"Invalid Gaussian opacity at point {index}")
    return {name: data_type for name, (data_type, _) in specs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply", type=Path, help="Idol PLY to import")
    parser.add_argument("--expected-count", type=int, help="Required number of Gaussian points")
    parser.add_argument("--blend", type=Path, help="Optional output .blend path")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    if not bpy.app.background or bpy.app.version < (5, 3, 0):
        raise RuntimeError("Run this script with Blender 5.3 or later in background mode")
    if bpy.data.filepath:
        raise RuntimeError("Start Blender without an existing .blend file")

    path = args.ply.resolve(strict=True)
    count = ply_vertex_count(path)
    if args.expected_count is not None and count != args.expected_count:
        raise RuntimeError(f"PLY declares {count} points, expected {args.expected_count}")

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    # Idol writes (x, -z, y) relative to its internal Z-up coordinate system.
    forward_axis, up_axis = "Z", "NEGATIVE_Y"
    result = bpy.ops.wm.ply_import(
        filepath=str(path),
        forward_axis=forward_axis,
        up_axis=up_axis,
        global_scale=1.0,
        use_scene_unit=False,
        merge_verts=False,
        import_attributes=True,
    )
    objects = list(bpy.context.scene.objects)
    if result != {"FINISHED"} or len(objects) != 1:
        raise RuntimeError(f"PLY import returned {result} and produced {len(objects)} objects")
    obj = objects[0]
    if obj.type != "POINTCLOUD" or obj.data.type != "GAUSSIAN_SPLAT":
        raise RuntimeError(f"Expected a native Gaussian Splat, got {obj.type}")
    if len(obj.data.points) != count:
        raise RuntimeError(f"Blender imported {len(obj.data.points)} points, expected {count}")
    attributes = validate_attributes(obj.data, count)
    bpy.context.view_layer.update()
    axis_sample = obj.matrix_world @ Vector((1.0, 2.0, 3.0))
    if (axis_sample - Vector((1.0, 3.0, -2.0))).length > 1e-5:
        raise RuntimeError(f"Unexpected Idol-to-Blender axis transform {tuple(axis_sample)}")

    blend = None
    if args.blend is not None:
        blend = args.blend.resolve()
        if blend.suffix.lower() != ".blend":
            raise ValueError(f"Expected a .blend output path, got {blend}")
        blend.parent.mkdir(parents=True, exist_ok=True)
        if bpy.ops.wm.save_as_mainfile(filepath=str(blend)) != {"FINISHED"}:
            raise RuntimeError(f"Saving {blend} did not finish")
    print(json.dumps({
        "blender": bpy.app.version_string,
        "type": obj.data.type,
        "points": count,
        "attributes": attributes,
        "forward_axis": forward_axis,
        "up_axis": up_axis,
        "blend": str(blend) if blend else None,
    }))


if __name__ == "__main__":
    main()
