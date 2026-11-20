"""Main-thread orchestration for the isolated Blender generation worker."""

import atexit
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from uuid import uuid4

import bpy
from bpy.app.handlers import persistent
from mathutils import Vector

from .jobs import Job
from .pipeline import BAKE_RESOLUTIONS, is_gaussian


TRANSFER_COLLECTION = "IS3D Transfer"
_active = None
_log = logging.getLogger(__name__)


@dataclass
class ActiveJob:
    supervisor: Job
    scene_id: int
    cursor: tuple
    kind: str
    cancel_requested: bool = False


def is_busy():
    return _active is not None


def native_assets():
    """The bundled engine binary and weights, validated up front so the worker cannot fail late."""
    from .native_engine import check_assets

    return check_assets()


def start_generation(scene):
    global _active
    if is_busy():
        raise RuntimeError("已有生成任务在后台运行")
    settings = scene.is3d_mesh_options
    if not settings.reference_image:
        raise ValueError("请选择参考图")
    image = Path(bpy.path.abspath(settings.reference_image)).expanduser().resolve()
    if not image.is_file():
        raise ValueError(f"找不到参考图 {image}")
    dll, weights = native_assets()
    worker = Path(__file__).resolve().with_name("worker.py")
    if not worker.is_file() or not bpy.app.binary_path:
        raise RuntimeError("无法定位后台 Blender 或生成脚本")
    resolution = int(settings.resolution or BAKE_RESOLUTIONS[-1])
    if settings.output_kind == "LOW" and resolution not in BAKE_RESOLUTIONS:
        raise ValueError("请选择 512、1024 或 2048 的贴图分辨率")

    if settings.output_directory:
        parent = Path(bpy.path.abspath(settings.output_directory)).expanduser().resolve()
    else:
        # Never fall back inside the add-on folder: an unsaved scene would otherwise scatter job
        # folders into the Blender scripts directory.
        fallback = Path(bpy.data.filepath).parent if bpy.data.filepath else Path.home() / "Documents"
        parent = fallback / "IS3D assets"
    output = parent / f"is3d-{uuid4().hex[:10]}"
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "image": str(image),
        "output_dir": str(output),
        "output_kind": settings.output_kind,
        "high_method": settings.high_method,
        "low_method": settings.low_method,
        "voxel_size": settings.voxel_size,
        "target_triangles": settings.triangles,
        "resolution": resolution,
        "engine_dll": str(dll),
        "engine_weights": str(weights),
        "seed": settings.seed,
        "steps": settings.steps,
        "points": int(settings.gaussian_count),
        "mode": settings.image_mode,
        "remove_background": bool(settings.remove_background),
    }
    job_path = output / "job.json"
    job_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    supervisor = Job(bpy.app.binary_path, str(worker), str(job_path), str(output))
    _active = ActiveJob(supervisor, scene.session_uid, tuple(scene.cursor.location), settings.output_kind)
    try:
        supervisor.start()
        bpy.app.timers.register(poll_generation, first_interval=0.1)
    except Exception:
        supervisor.close()
        _active = None
        raise
    settings.job_progress = 0.0
    settings.job_message = "正在启动后台 Blender"
    settings.last_job_dir = str(output)
    settings.job_state = "RUNNING"
    return output


def request_cancel():
    if _active:
        _active.cancel_requested = True
        _active.supervisor.cancel()


def _target_scene(scene_id):
    return next((scene for scene in bpy.data.scenes if scene.session_uid == scene_id), None)


def displayed_status(scene):
    if _active is not None and (target := _target_scene(_active.scene_id)) is not None:
        return target.is3d_mesh_options
    return scene.is3d_mesh_options


def append_transfer(path, scene, cursor, kind):
    with bpy.data.libraries.load(str(path), link=False) as (contents, selected):
        if TRANSFER_COLLECTION not in contents.collections:
            raise ValueError("后台产物缺少可导入的资产集合")
        selected.collections = [TRANSFER_COLLECTION]
    collection = selected.collections[0]
    if collection is None:
        raise ValueError("无法读取后台产物集合")
    try:
        objects = list(collection.all_objects)
        if len(objects) != 1:
            raise ValueError("单张参考图任务应返回一个物体")
        obj = objects[0]
        if obj.parent is not None:
            raise ValueError("后台产物应以独立的物体原点定位")
        if (kind == "SPLAT" and not is_gaussian(obj)) or (kind != "SPLAT" and obj.type != "MESH"):
            raise ValueError("后台产物与请求的物体类型不一致")
        collection.name = "IS3D 生成资产"
        scene.collection.children.link(collection)
        obj.location = Vector(cursor)
        obj.hide_viewport = False
        obj.hide_render = False
    except Exception:
        for obj in list(collection.all_objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(collection)
        raise
    if bpy.context.scene == scene and bpy.context.mode == "OBJECT":
        layer = bpy.context.view_layer
        layer.update()
        if obj.name in layer.objects:
            for selected_object in bpy.context.selected_objects:
                selected_object.select_set(False)
            obj.hide_set(False)
            obj.select_set(True)
            layer.objects.active = obj
    return obj


def _refresh_library():
    # Job folders appear as soon as a run is complete, so the panel list must drop its cache;
    # the worker writes the thumbnail next to the asset.
    from . import library

    library.invalidate()


def _redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def poll_generation():
    global _active
    active = _active
    if active is None:
        return None
    scene = _target_scene(active.scene_id)
    if scene is None:
        request_cancel()
    settings = scene.is3d_mesh_options if scene is not None else None
    if active.cancel_requested and settings is not None:
        settings.job_message = "正在取消后台任务"

    for event in active.supervisor.poll_events():
        kind = event["kind"]
        if kind == "progress":
            if settings is not None and not active.cancel_requested:
                settings.job_progress = max(settings.job_progress, min(float(event["percent"]) / 100, 0.99))
                settings.job_message = event["message"]
            continue
        if settings is not None:
            try:
                if active.cancel_requested or kind == "cancelled":
                    settings.job_state = "CANCELLED"
                    settings.job_message = "任务已取消"
                elif kind == "finished":
                    obj = append_transfer(event["result_path"], scene, active.cursor, active.kind)
                    settings.job_progress = 1.0
                    settings.job_state = "FINISHED"
                    settings.job_message = f"已导入 {obj.name}"
                    _refresh_library()
                else:
                    settings.job_state = "ERROR"
                    settings.job_message = event.get("message", "后台生成失败，请查看任务日志")
            except Exception as error:
                _log.exception("IS3D transfer failed")
                settings.job_state = "ERROR"
                settings.job_message = f"导入失败 · {error}"
        _active = None
        _redraw()
        return None
    _redraw()
    return 0.2


def shutdown_jobs():
    global _active
    active, _active = _active, None
    if active is not None:
        active.supervisor.close(timeout=1.0)


@persistent
def _before_file_load(_unused):
    global _active
    active, _active = _active, None
    if active is not None:
        active.supervisor.close()
    if bpy.app.timers.is_registered(poll_generation):
        bpy.app.timers.unregister(poll_generation)


@persistent
def _reset_progress(_unused):
    # Enabling the add-on during startup can run this while bpy.data is still restricted.
    scenes = getattr(bpy.data, "scenes", None)
    if scenes is None:
        return
    for scene in scenes:
        settings = scene.is3d_mesh_options
        # The v0.2 Blender enum stored 4096 as item 3. Keep older scenes editable.
        if settings.get("resolution") == 3:
            settings.resolution = str(BAKE_RESOLUTIONS[-1])
        for name in ("job_progress", "job_message", "job_state", "last_job_dir"):
            settings.property_unset(name)


def register_handlers():
    bpy.app.handlers.load_pre.append(_before_file_load)
    bpy.app.handlers.load_post.append(_reset_progress)
    _reset_progress(None)
    atexit.register(shutdown_jobs)


def unregister_handlers():
    _before_file_load(None)
    bpy.app.handlers.load_pre.remove(_before_file_load)
    bpy.app.handlers.load_post.remove(_reset_progress)
    atexit.unregister(shutdown_jobs)
