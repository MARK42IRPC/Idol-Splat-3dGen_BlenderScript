"""Render and check two orthographic views of an imported Blender Gaussian splat."""

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np


def validate_render(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = image.size
        pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        alpha = pixels[3::4]
        coverage = float(np.count_nonzero(alpha > 0.05) / alpha.size)
        if not 0.01 < coverage < 0.98:
            raise RuntimeError(f"Unexpected alpha coverage in {path.name} ({coverage:.1%})")
        return coverage
    finally:
        bpy.data.images.remove(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output directory for two RGBA PNGs")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    if not bpy.app.background or bpy.app.version < (5, 3, 0):
        raise RuntimeError("Run with Blender 5.3 or later in background mode")

    scene = bpy.context.scene
    clouds = [obj for obj in scene.objects if obj.type == "POINTCLOUD" and obj.data.type == "GAUSSIAN_SPLAT"]
    if len(clouds) != 1:
        raise RuntimeError(f"Expected one native Gaussian splat object, found {len(clouds)}")

    obj = clouds[0]
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(corners, Vector()) / len(corners)
    radius = max((corner - center).length for corner in corners)
    if not radius > 0:
        raise RuntimeError("Splat bounds are empty")

    camera = bpy.data.cameras.new("Validation Camera")
    camera.type = "ORTHO"
    camera.clip_end = max(10.0, 10.0 * radius)
    camera_obj = bpy.data.objects.new(camera.name, camera)
    scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.display.shading.light = "FLAT"
    scene.display.shading.show_shadows = False
    scene.display.shading.show_cavity = False
    scene.display.shading.show_object_outline = False

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    coverage = {}
    views = {"view_a": Vector((1.0, -2.0, 1.0)), "view_b": Vector((-1.0, 2.0, 1.0))}
    for name, direction in views.items():
        rotation = (-direction).to_track_quat("-Z", "Y")
        camera_obj.location = center + direction.normalized() * max(1.0, 3.0 * radius)
        camera_obj.rotation_euler = rotation.to_euler()
        projected_corners = [rotation.inverted() @ (corner - center) for corner in corners]
        camera.ortho_scale = 2.25 * max(max(abs(corner.x), abs(corner.y)) for corner in projected_corners)
        path = output / f"{name}.png"
        scene.render.filepath = str(path)
        if bpy.ops.render.render(write_still=True) != {"FINISHED"}:
            raise RuntimeError(f"Rendering {path} did not finish")
        coverage[name] = validate_render(path)

    print(json.dumps({"engine": scene.render.engine, "opacity_coverage": coverage}))


if __name__ == "__main__":
    main()
