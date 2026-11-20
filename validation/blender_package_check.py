"""Drive the packaged add-on end to end inside Blender and render the imported asset."""

import argparse
import json
import sys
import time
from pathlib import Path

import addon_utils
import bpy
import numpy as np
from mathutils import Vector

TRANSFER_COLLECTION = "IS3D 生成资产"


def arguments():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--kind", default="SPLAT")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--points", type=int, default=32768)
    parser.add_argument("--triangles", type=int, default=20000)
    parser.add_argument("--resolution", default="1024")
    parser.add_argument("--voxel-size", dest="voxel_size", type=float, default=0.006)
    parser.add_argument("--matte", dest="matte", action="store_true", default=True)
    parser.add_argument("--no-matte", dest="matte", action="store_false")
    return parser.parse_args(argv)


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def render_views(obj, output):
    scene = bpy.context.scene
    splat = obj.type == "POINTCLOUD"
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(corners, Vector()) / len(corners)
    radius = max((corner - center).length for corner in corners)
    check(radius > 0, "asset bounds are empty")
    camera = bpy.data.cameras.new("Validation Camera")
    camera.type = "ORTHO"
    camera.clip_end = max(10.0, 10.0 * radius)
    camera_obj = bpy.data.objects.new(camera.name, camera)
    scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj
    # EEVEE shows the authored look for both kinds: splats carry radiance colours and the mesh
    # materials are emission (vertex colour) or baked textures, so one sun lamp is enough.
    scene.render.engine = "BLENDER_EEVEE"
    if not splat:
        lamp = bpy.data.objects.new("Validation Sun", bpy.data.lights.new("Validation Sun", "SUN"))
        lamp.data.energy = 3.0
        lamp.rotation_euler = (0.85, 0.1, 0.6)
        scene.collection.objects.link(lamp)
    size = 512 if splat else 768
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    coverage = {}
    views = {"view_a": Vector((1.0, -2.0, 1.0)), "view_b": Vector((-1.0, 2.0, 1.0))}
    for name, direction in views.items():
        rotation = (-direction).to_track_quat("-Z", "Y")
        camera_obj.location = center + direction.normalized() * max(1.0, 3.0 * radius)
        camera_obj.rotation_euler = rotation.to_euler()
        projected = [rotation.inverted() @ (corner - center) for corner in corners]
        camera.ortho_scale = 2.3 * max(max(abs(corner.x), abs(corner.y)) for corner in projected)
        check(camera.ortho_scale > 0, f"degenerate framing for {name}")
        path = output / f"{name}.png"
        scene.render.filepath = str(path)
        check(bpy.ops.render.render(write_still=True) == {"FINISHED"}, f"render {path} failed")
        image = bpy.data.images.load(str(path), check_existing=False)
        try:
            pixels = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
            image.pixels.foreach_get(pixels)
        finally:
            bpy.data.images.remove(image)
        alpha = pixels[3::4]
        coverage[name] = float(np.count_nonzero(alpha > 0.05) / alpha.size)
        check(0.01 < coverage[name] < 0.98, f"unexpected coverage in {path.name}: {coverage[name]:.1%}")
        if splat:
            rgb = pixels.reshape(-1, 4)[:, :3]
            visible = rgb[alpha > 0.05]
            spread = float(visible.std()) if visible.size else 0.0
            check(spread > 0.01, f"rendered splat in {path.name} looks flat (colour std {spread:.4f})")
    return coverage


def main():
    args = arguments()
    check(bpy.app.background and bpy.app.version >= (5, 3, 0), "Blender 5.3+ background only")
    check(addon_utils.enable("is3d", default_set=True, persistent=False),
          "the packaged add-on could not be enabled")
    import is3d
    from is3d import async_ui

    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Drop the factory startup cube so it cannot fill the validation frame.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    scene = bpy.context.scene
    settings = scene.is3d_mesh_options
    settings.reference_image = str(args.image.resolve())
    settings.output_kind = args.kind
    settings.output_directory = str(output)
    settings.steps = args.steps
    settings.seed = 42
    settings.gaussian_count = str(args.points)
    settings.triangles = args.triangles
    settings.resolution = args.resolution
    settings.voxel_size = args.voxel_size
    settings.image_mode = "synthetic"
    settings.remove_background = args.matte

    started = time.time()
    async_ui.start_generation(scene)
    events, seen, last = 0, set(), ""
    while async_ui.is_busy():
        async_ui.poll_generation()
        if settings.job_message != last:
            last = settings.job_message
            seen.add(last)
        events += 1
        check(time.time() - started < 3600, "generation timed out")
        time.sleep(0.15)
    seconds = time.time() - started
    check(settings.job_state == "FINISHED", f"generation ended as {settings.job_state}: {settings.job_message}")

    collection = bpy.data.collections.get(TRANSFER_COLLECTION)
    check(collection is not None, "the add-on did not append its asset collection")
    objects = [obj for obj in collection.all_objects]
    check(len(objects) == 1, f"expected one asset object, found {len(objects)}")
    obj = objects[0]
    result = {
        "blender": bpy.app.version_string,
        "addon_version": ".".join(str(part) for part in is3d.bl_info["version"]),
        "kind": args.kind, "object": obj.name, "object_type": obj.type,
        "location": list(obj.location), "seconds": round(seconds, 2),
        "distinct_messages": len(seen), "polls": events,
        "job_folder": settings.last_job_dir,
        "remove_background": settings.remove_background,
    }
    if obj.type == "POINTCLOUD":
        check(obj.data.type == "GAUSSIAN_SPLAT", "imported point cloud is not a gaussian splat")
        result["gaussian_points"] = len(obj.data.points)
        check(len(obj.data.points) == args.points, "gaussian count mismatch")
        result["axis_mapping"] = [float(v) for v in obj.matrix_world.to_translation()]
        result["render_coverage"] = render_views(obj, output)
    else:
        check(obj.type == "MESH", f"unexpected object type {obj.type}")
        result["vertices"] = len(obj.data.vertices)
        result["polygons"] = len(obj.data.polygons)
        result["materials"] = [slot.material.name for slot in obj.material_slots if slot.material]
        check(len(obj.data.polygons) > 0, "mesh has no faces")
        result["render_coverage"] = render_views(obj, output)

    report_path = Path(settings.last_job_dir) / "inference" / "report.json"
    check(report_path.is_file(), f"engine report missing at {report_path}")
    result["engine"] = json.loads(report_path.read_text(encoding="utf-8"))

    scene.render.filepath = ""
    blend = output / "scene_with_asset.blend"
    check(bpy.ops.wm.save_as_mainfile(filepath=str(blend)) == {"FINISHED"}, "could not save the scene")
    result["scene"] = str(blend)
    (output / "integration.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("IS3D_CHECK " + json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
