"""Background Blender entry point for a single local IS3D image job."""

import argparse
import json
from pathlib import Path
import sys

import bpy


TRANSFER_COLLECTION = "IS3D Transfer"
TRANSFER_FILE = "asset_transfer.blend"


class Progress:
    def __init__(self):
        self.percent = 0

    def emit(self, percent, message):
        self.percent = max(self.percent, int(percent))
        data = {"percent": self.percent, "message": str(message)}
        print("IS3D_PROGRESS " + json.dumps(data, ensure_ascii=True), flush=True)

    def mesh_line(self, message):
        stages = {"[1/4]": 72, "[2/4]": 79, "[3/4]": 84, "[4/4]": 88}
        self.emit(stages.get(message[:5], self.percent), message)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True, help="Absolute path to one IS3D job JSON file")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def _absolute_path(value, name):
    if not value:
        raise ValueError(f"作业缺少路径 {name}")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} 必须是绝对路径")
    return path.resolve()


def _input_file(job, name):
    path = _absolute_path(job.get(name), name)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {name} 文件 {path}")
    return path


def _image_rgba(path):
    """Decode any image Blender understands into an sRGB RGBA float32 array (top row first)."""
    import numpy as np

    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = image.size
        if width < 8 or height < 8:
            raise ValueError(f"参考图尺寸过小：{width} x {height}")
        # Read the stored values verbatim: the reference pipeline feeds sRGB bytes/255 into the
        # encoders, so the picture must not be colour managed on the way in.
        image.colorspace_settings.name = "Non-Color"
        buffer = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(buffer)
        rgba = np.ascontiguousarray(buffer.reshape(height, width, 4)[::-1])
    finally:
        bpy.data.images.remove(image)
    return rgba, width, height


def _run_native(job, output, progress):
    """Run the bundled CUDA-driver engine; it is the only inference dependency we ship."""
    from is3d.native_engine import ABI, Engine, NativeEngineError, budget_report, check_assets

    if job.get("source_ply"):
        progress.emit(67, "使用已有高斯 PLY 进行本地验证")
        return _input_file(job, "source_ply")

    image = _input_file(job, "image")
    dll, weights = check_assets(job.get("engine_dll"), job.get("engine_weights"))
    rgba, width, height = _image_rgba(image)
    directory = output / "inference"
    directory.mkdir(parents=True, exist_ok=True)
    margin = 1.7 if job.get("mode", "synthetic") == "real" else 1.2
    progress.emit(1, "正在启动本地推理引擎")
    with Engine(dll, weights) as engine:
        info = engine.info()
        if info["free_mib"] < 2600.0:
            raise NativeEngineError(
                f"可用显存只有 {info['free_mib']:.0f} MiB，至少需要 2600 MiB。"
                "请关闭其他占用显卡的程序（浏览器视频、游戏、其它生成任务）后重试")
        engine.set_crop_margin(margin)
        remove_background = bool(job.get("remove_background", True))
        engine.set_remove_background(remove_background)
        engine.reset_peak()
        stats = engine.generate(
            rgba, width, height, directory / "asset.ply", directory / "camera.json",
            seed=int(job.get("seed", 42)), steps=int(job.get("steps", 50)),
            points=int(job.get("points", 262144)), progress=progress.emit)
        budget = budget_report(stats["peak_mib"])
        report = {
            "engine": "idol_engine (native CUDA driver)", "abi": ABI, "image": str(image),
            "image_size": [width, height], "crop_margin": margin,
            "remove_background": remove_background,
            "device": info["device"], "compute_capability": info["compute_capability"],
            "seconds": round(stats["seconds"], 2), "peak_mib": round(stats["peak_mib"], 1),
            "budget": budget, "steps": int(job.get("steps", 50)),
            "points": int(job.get("points", 262144)), "seed": int(job.get("seed", 42)),
        }
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    result = directory / "asset.ply"
    if not result.is_file() or not result.stat().st_size:
        raise RuntimeError("推理没有生成有效的高斯文件")
    if budget["status"] == "error":
        raise NativeEngineError(budget["message"])
    if budget["status"] == "warning":
        progress.emit(progress.percent, budget["message"])
    return result


def _import_gaussian(path):
    before = set(bpy.context.scene.objects)
    status = bpy.ops.wm.ply_import(
        filepath=str(path), forward_axis="Z", up_axis="NEGATIVE_Y",
        global_scale=1.0, use_scene_unit=False, merge_verts=False, import_attributes=True,
    )
    imported = set(bpy.context.scene.objects) - before
    if status != {"FINISHED"} or len(imported) != 1:
        raise RuntimeError(f"无法导入 IS3D 高斯 PLY {path}")
    source = imported.pop()
    if source.type != "POINTCLOUD" or source.data.type != "GAUSSIAN_SPLAT" or not len(source.data.points):
        raise ValueError("导入结果缺少原生高斯点云数据")
    bpy.context.view_layer.update()
    return source


def _high_material(obj):
    colors = obj.data.color_attributes
    colors.active_color = colors["IS3DColor"]
    material = bpy.data.materials.new("IS3D 高模颜色")
    material.use_nodes = True
    shader = material.node_tree
    shader.nodes.clear()
    color = shader.nodes.new("ShaderNodeVertexColor")
    color.layer_name = "IS3DColor"
    color.location = (-320, 0)
    emission = shader.nodes.new("ShaderNodeEmission")
    emission.location = (-70, 0)
    output = shader.nodes.new("ShaderNodeOutputMaterial")
    output.location = (160, 0)
    shader.links.new(color.outputs["Color"], emission.inputs["Color"])
    shader.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    # The reconstruction can leave an empty slot at index 0 while every polygon points at it, so
    # replace the whole slot list instead of appending behind the stale entry.
    mesh = obj.data
    mesh.materials.clear()
    mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.material_index = 0


def _export_collection(collection, output):
    destination = output / TRANSFER_FILE
    temporary = output / "asset_transfer.partial.blend"
    try:
        bpy.data.libraries.write(str(temporary), {collection}, fake_user=True, compress=True)
        if not temporary.is_file() or not temporary.stat().st_size:
            raise RuntimeError("Blender 没有生成临时资产文件")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_job(job, progress):
    from is3d.high_mesh import build_high
    from is3d.pipeline import pack_result_images, run_pipeline, transfer_color

    kind = job.get("output_kind")
    if kind not in {"SPLAT", "HIGH", "LOW"}:
        raise ValueError(f"无法识别的 IS3D 资产类型 {kind!r}")
    output = _absolute_path(job.get("output_dir"), "output_dir")
    output.mkdir(parents=True, exist_ok=True)
    if (output / TRANSFER_FILE).exists():
        raise FileExistsError(f"目录已经包含完成的 IS3D 资产 {output}")

    ply = _run_native(job, output, progress)
    progress.emit(70, "正在导入 Blender 原生高斯点云")
    source = _import_gaussian(ply)
    collection = bpy.data.collections.new(TRANSFER_COLLECTION)
    bpy.context.scene.collection.children.link(collection)

    if kind == "SPLAT":
        result = source
    else:
        voxel_size = float(job.get("voxel_size", 0.006))
        high_method = job.get("high_method", "AUTO")
        if kind == "HIGH":
            progress.emit(72, "正在去噪和重建高模")
            result, geometry_report = build_high(source, collection, method=high_method, voxel_size=voxel_size,
                                                 progress=progress.mesh_line)
            progress.emit(83, "正在将高斯颜色转移到高模")
            color_report = transfer_color(source, result, geometry_report["voxel_size"], progress.mesh_line)
            _high_material(result)
            details = {"status": "passed", "high": geometry_report, "color": color_report}
            (output / "report.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
        else:
            result, _ = run_pipeline(
                source, output, high_method=high_method, low_method=job.get("low_method", "AUTO"),
                voxel_size=voxel_size, target_triangles=int(job.get("target_triangles", 20000)),
                resolution=int(job.get("resolution", 2048)), progress=progress.mesh_line,
            )
            pack_result_images(result)

    if collection.objects.get(result.name) is None:
        collection.objects.link(result)
    for previous in tuple(result.users_collection):
        if previous != collection:
            previous.objects.unlink(result)
    progress.emit(96, "正在保存可附加的 Blender 资产")
    _export_collection(collection, output)
    progress.emit(100, "Blender 资产已生成")


def main():
    if not bpy.app.background or bpy.app.version < (5, 3, 0):
        raise RuntimeError("需要以后台模式运行 Blender 5.3 或更新版本")
    if bpy.data.filepath or bpy.data.collections.get(TRANSFER_COLLECTION):
        raise RuntimeError("请使用工厂启动模式运行独立的 IS3D 后台进程")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    job_path = _absolute_path(_arguments().job, "job")
    progress = Progress()
    try:
        job = json.loads(job_path.read_text(encoding="utf-8-sig"))
        run_job(job, progress)
    except Exception as error:
        progress.emit(progress.percent, f"IS3D 生成失败。{error}")
        raise


if __name__ == "__main__":
    main()
