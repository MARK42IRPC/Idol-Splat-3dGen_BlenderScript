"""Convert a Blender 5.3 IS3D Gaussian Point Cloud to a textured static asset."""

import json
from pathlib import Path
from time import perf_counter

import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree


HIGH_METHODS = (
    ("AUTO", "自动推荐", "实体道具优先用稳定的 Blender 原生体素重建"),
    ("VOXEL", "Blender 体素", "原生球形体素核，适合快速重建和闭环验证"),
    ("ELLIPSOID", "各向异性高斯", "用三轴尺度和旋转构建密度场，着重保留细长高斯的形状"),
    ("SDF", "去噪 + SDF 拟合", "用窄带有符号距离场重提取表面，缓解体素阶梯"),
)
LOW_METHODS = (
    ("AUTO", "自动推荐", "静态资源默认使用 QEM，按三角面预算减面"),
    ("QEM", "QEM 减面", "保留原表面并按目标三角面数减面"),
    ("QUADRIFLOW", "QuadriFlow", "先提取四边面拓扑，再三角化用于烘焙"),
    ("VACUUM", "收缩包覆（实验）", "用压力式几何收缩逼近表面，深凹和通孔需复核"),
)
TEXTURE_FILES = ("base_color.png", "normal.png")
BAKE_RESOLUTIONS = (512, 1024, 2048)


def is_gaussian(obj):
    return obj is not None and obj.type == "POINTCLOUD" and obj.data.type == "GAUSSIAN_SPLAT"


def transfer_color(source, high, voxel_size, progress=print):
    """Sample linear SH0 color on the high surface with local Gaussian weights."""
    from .high_mesh import read_gaussians

    points, scales, quaternions, radiance = read_gaussians(source)
    eligible = (radiance[:, 3] >= 0.001) & np.isfinite(radiance).all(axis=1)
    if not eligible.any():
        raise ValueError("Gaussian input has no visible color samples")
    points, scales, quaternions, radiance = (
        array[eligible] for array in (points, scales, quaternions, radiance)
    )

    # PLY degree-zero SH represents the captured sRGB appearance.
    srgb = np.clip(0.5 + np.float32(0.28209479177387814) * radiance[:, :3], 0, 1)
    linear_rgb = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    alpha = radiance[:, 3]
    quaternions /= np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-8)

    tree = KDTree(len(points))
    for index, point in enumerate(points):
        tree.insert(point, index)
    tree.balance()

    mesh = high.data
    vertices = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    normals = np.empty_like(vertices)
    mesh.vertices.foreach_get("co", vertices.ravel())
    mesh.vertex_normals.foreach_get("vector", normals.ravel())
    high_to_world = np.asarray(high.matrix_world, dtype=np.float32)
    vertices = vertices @ high_to_world[:3, :3].T + high_to_world[:3, 3]
    normals = normals @ np.linalg.inv(high_to_world[:3, :3])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    world_to_source = np.asarray(source.matrix_world.inverted().to_3x3(), dtype=np.float32)
    band = voxel_size * np.linalg.norm(world_to_source, axis=1).mean()
    safe_scales = np.clip(scales, band * 0.8, band * 4)

    rgba = np.ones((len(vertices), 4), dtype=np.float32)
    nearest = np.empty(len(vertices), dtype=np.float32)
    count = min(16, len(points))
    for start in range(0, len(vertices), 4096):
        window = slice(start, start + 4096)
        indices = np.array([[index for _, index, _ in tree.find_n(vertex, count)]
                            for vertex in vertices[window]], dtype=np.intp)
        offsets = points[indices] - vertices[window, None, :]
        nearest[window] = np.linalg.norm(offsets[:, 0], axis=1)
        local = offsets @ world_to_source.T
        rotation = quaternions[indices]
        cross = np.cross(rotation[..., 1:], local)
        oriented = local - 2 * rotation[..., :1] * cross + 2 * np.cross(rotation[..., 1:], cross)
        squared_distance = np.sum((oriented / safe_scales[indices]) ** 2, axis=2)
        # Samples beyond the outward surface receive less weight at thin edges.
        signed_distance = np.sum(offsets * normals[window, None, :], axis=2)
        outward = np.maximum(signed_distance - voxel_size, 0) / voxel_size
        log_weights = np.log(alpha[indices]) - 0.5 * squared_distance - outward
        weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
        rgba[window, :3] = np.sum(linear_rgb[indices] * weights[..., None], axis=1) / weights.sum(
            axis=1, keepdims=True
        )

    old = mesh.color_attributes.get("IS3DColor")
    if old is not None:
        mesh.color_attributes.remove(old)
    color = mesh.color_attributes.new(name="IS3DColor", type="FLOAT_COLOR", domain="POINT")
    color.data.foreach_set("color", rgba.ravel())
    mesh.update()
    progress(f"颜色转移完成，共 {len(vertices):,} 个高模顶点")
    return {
        "source_gaussians": int(len(points)),
        "colored_vertices": int(len(vertices)),
        "nearest_distance_p95": float(np.quantile(nearest, 0.95)),
        "fraction_within_four_voxels": float(np.mean(nearest <= voxel_size * 4)),
    }


def activate_result(source, high, low):
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in (source, high):
        obj.hide_set(True)
        obj.hide_render = True
    low.hide_set(False)
    low.hide_render = False
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    bpy.context.view_layer.update()
    center = sum((low.matrix_world @ Vector(corner) for corner in low.bound_box), Vector()) / 8
    distance = max(low.dimensions.length * 1.5, 0.1)
    if bpy.context.screen is not None:
        for area in bpy.context.screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                space.region_3d.view_location = center
                space.region_3d.view_distance = distance
                if space.shading.type == "SOLID":
                    space.shading.color_type = "TEXTURE"


def pack_result_images(low):
    for material in low.data.materials:
        if material is None or material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None and node.image.packed_file is None:
                node.image.pack()


def run_pipeline(source, output_dir, *, high_method="AUTO", low_method="AUTO",
                 voxel_size=0.006, target_triangles=20000, resolution=2048, progress=print):
    from .bake import unwrap_and_bake
    from .high_mesh import build_high
    from .low_mesh import build_low

    if bpy.app.version < (5, 3, 0) or bpy.context.mode != "OBJECT":
        raise ValueError("使用 Blender 5.3 或更新版本，并切换到物体模式")
    if not is_gaussian(source):
        raise ValueError("请选择一个原生 Gaussian Splat 点云")
    if high_method not in {item[0] for item in HIGH_METHODS}:
        raise ValueError(f"未知高模算法 {high_method}")
    if low_method not in {item[0] for item in LOW_METHODS}:
        raise ValueError(f"未知低模算法 {low_method}")
    if not 0 < voxel_size or target_triangles < 32 or resolution < 128:
        raise ValueError("体素尺寸须为正，低模至少 32 面，贴图至少 128 像素")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if any((output / name).exists() for name in (*TEXTURE_FILES, report_path.name)):
        raise FileExistsError(f"目标目录已有管线结果 {output}")

    collection = bpy.data.collections.new("IS3D Mesh Asset")
    bpy.context.scene.collection.children.link(collection)
    start = perf_counter()
    report = {
        "blender": bpy.app.version_string,
        "source": source.name,
        "gaussians": len(source.data.points),
        "voxel_size": voxel_size,
        "target_triangles": target_triangles,
        "texture_resolution": resolution,
        "high_requested": high_method,
        "low_requested": low_method,
    }
    stage = "high"
    try:
        progress("[1/4] 去噪与高模重建")
        high, report["high"] = build_high(source, collection, method=high_method, voxel_size=voxel_size,
                                         progress=progress)
        stage = "color"
        progress("[2/4] 将高斯颜色转移到高模")
        report["color"] = transfer_color(source, high, voxel_size, progress)
        stage = "low"
        progress("[3/4] 生成低模")
        low, report["low"] = build_low(high, collection, method=low_method, target_triangles=target_triangles,
                                      progress=progress)
        stage = "bake"
        progress("[4/4] 展开 UV 与 Cycles 烘焙")
        report["bake"] = unwrap_and_bake(high, low, output, resolution=resolution, progress=progress)
        pack_result_images(low)
        activate_result(source, high, low)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", failed_stage=stage, error=str(error))
        # Partial geometry remains inspectable in the generated collection.
        raise
    finally:
        report["seconds"] = round(perf_counter() - start, 3)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return low, report
