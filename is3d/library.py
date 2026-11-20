"""Generated asset library: thumbnails, listing, import and drag placement.

Every job writes its result into a folder next to ``job.json``:

    is3d-<id>/job.json              job configuration (kind, image, budget)
    is3d-<id>/asset_transfer.blend  the finished object, ready to append
    is3d-<id>/thumb.png             384 px preview rendered by the worker
    is3d-<id>/inference/            engine artefacts (ply, camera, report)

The panel lists those folders so a previous result can be dropped into any
scene without running the model again.
"""

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import time

import bpy

from . import async_ui

TRANSFER_FILE = "asset_transfer.blend"
THUMB_FILE = "thumb.png"
CACHE_DIR = ".is3d-thumbs"
MAX_ITEMS = 60
KIND_LABELS = {"SPLAT": "高斯点云", "HIGH": "高模", "LOW": "低模"}
ICON_BY_KIND = {"SPLAT": "POINTCLOUD_DATA", "HIGH": "MESH_DATA", "LOW": "MESH_GRID"}

_previews = None
_loaded = {}
_cache = {"key": None, "items": ()}


@dataclass(frozen=True)
class AssetInfo:
    folder: str
    title: str
    kind: str
    detail: str
    created: float
    thumb: str

    @property
    def kind_label(self):
        return KIND_LABELS.get(self.kind, self.kind)

    @property
    def icon(self):
        return ICON_BY_KIND.get(self.kind, "OBJECT_DATA")


# ---------------------------------------------------------------------- paths
def library_root(scene):
    settings = scene.is3d_mesh_options if scene is not None else None
    if settings is not None and settings.output_directory:
        return Path(bpy.path.abspath(settings.output_directory)).expanduser()
    if bpy.data.filepath:
        return Path(bpy.data.filepath).parent / "IS3D assets"
    return Path.home() / "Documents" / "IS3D assets"


def _cache_dir(root):
    path = Path(root) / CACHE_DIR
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return path


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


# ----------------------------------------------------------------- thumbnails
def _thumbnail_of(source, target, size=256):
    """Downscale any Blender-readable image into a small PNG used by the panel."""
    source, target = Path(source), Path(target)
    if not source.is_file():
        return None
    try:
        if target.is_file() and target.stat().st_mtime >= source.stat().st_mtime:
            return target
    except OSError:
        return None
    image = None
    try:
        image = bpy.data.images.load(str(source), check_existing=False)
        width, height = image.size
        if width < 2 or height < 2:
            return None
        scale = size / max(width, height)
        if scale < 1.0:
            image.scale(max(1, int(width * scale)), max(1, int(height * scale)))
        target.parent.mkdir(parents=True, exist_ok=True)
        image.filepath_raw = str(target)
        image.file_format = "PNG"
        image.save()
        return target
    except Exception:  # noqa: BLE001 - a thumbnail is cosmetic, never fatal
        return None
    finally:
        if image is not None:
            try:
                bpy.data.images.remove(image)
            except Exception:  # noqa: BLE001
                pass


def _reference_key(image):
    return hashlib.sha1(str(image).lower().encode("utf-8")).hexdigest()[:16]


def reference_thumbnail_path(scene):
    """Where the cached preview of the selected reference image lives (no disk writes)."""
    settings = scene.is3d_mesh_options
    if not settings or not settings.reference_image:
        return None
    root = _cache_dir(library_root(scene))
    if root is None:
        return None
    image = Path(bpy.path.abspath(settings.reference_image)).expanduser()
    return root / f"ref-{_reference_key(image)}.png"


def ensure_reference_thumbnail(scene):
    settings = scene.is3d_mesh_options
    if not settings or not settings.reference_image:
        return None
    target = reference_thumbnail_path(scene)
    if target is None:
        return None
    image = Path(bpy.path.abspath(settings.reference_image)).expanduser()
    return _thumbnail_of(image, target, size=320)


def reference_info(scene):
    """(width, height, megabytes) of the reference image, or None when it is missing."""
    settings = scene.is3d_mesh_options
    if not settings or not settings.reference_image:
        return None
    path = Path(bpy.path.abspath(settings.reference_image)).expanduser()
    if not path.is_file():
        return None
    image = None
    try:
        image = bpy.data.images.load(str(path), check_existing=False)
        width, height = image.size
    except Exception:  # noqa: BLE001
        return None
    finally:
        if image is not None:
            try:
                bpy.data.images.remove(image)
            except Exception:  # noqa: BLE001
                pass
    try:
        size = path.stat().st_size / (1024 * 1024)
    except OSError:
        size = 0.0
    return width, height, size


# --------------------------------------------------------------------- listing
def _folder_info(folder, cache):
    job = _read_json(folder / "job.json")
    kind = job.get("output_kind") or "SPLAT"
    if kind not in KIND_LABELS or not (folder / TRANSFER_FILE).is_file():
        return None
    image = Path(str(job.get("image") or ""))
    title = image.stem or folder.name
    detail = [KIND_LABELS[kind]]
    if kind == "SPLAT" and job.get("points"):
        detail.append(f"{int(job['points']):,} 点")
    elif kind != "SPLAT":
        if job.get("target_triangles"):
            detail.append(f"{int(job['target_triangles']):,} 面")
        if job.get("voxel_size"):
            detail.append(f"体素 {float(job['voxel_size']):.3f}")
    report = _read_json(folder / "inference" / "report.json")
    if report.get("seconds"):
        detail.append(f"{float(report['seconds']):.0f} 秒")
    try:
        created = (folder / TRANSFER_FILE).stat().st_mtime
    except OSError:
        created = 0.0
    thumb = folder / THUMB_FILE
    if not thumb.is_file() and cache is not None and image.is_file():
        thumb = _thumbnail_of(image, cache / f"ref-{_reference_key(image)}.png") or thumb
    return AssetInfo(
        folder=str(folder),
        title=title[:32],
        kind=kind,
        detail=" · ".join(detail),
        created=created,
        thumb=str(thumb) if Path(thumb).is_file() else "",
    )


def scan(scene, limit=MAX_ITEMS):
    """Finished jobs under the library root, newest first (cached until the folder changes)."""
    root = library_root(scene)
    try:
        stamp = (str(root), root.stat().st_mtime)
    except OSError:
        return []
    if _cache["key"] == stamp:
        return _cache["items"]
    cache = _cache_dir(root)
    items = []
    try:
        folders = sorted((path for path in root.iterdir()
                          if path.is_dir() and not path.name.startswith(".")),
                         key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        folders = []
    for folder in folders:
        try:
            info = _folder_info(folder, cache)
        except Exception:  # noqa: BLE001 - a half-written job must not break the panel
            info = None
        if info is not None:
            items.append(info)
            if len(items) >= limit:
                break
    _cache["key"] = stamp
    _cache["items"] = tuple(items)
    return _cache["items"]


def invalidate():
    _cache["key"] = None
    _cache["items"] = ()


def summary(scene):
    items = scan(scene)
    if not items:
        return "暂无历史资产"
    return f"{len(items)} 个 · {_ago(items[0].created)}"


def _ago(stamp):
    if not stamp:
        return "未知"
    seconds = max(0.0, time() - stamp)
    if seconds < 90:
        return f"{int(seconds)} 秒前"
    if seconds < 5400:
        return f"{int(seconds / 60)} 分钟前"
    if seconds < 172800:
        return f"{int(seconds / 3600)} 小时前"
    return f"{int(seconds / 86400)} 天前"


# ---------------------------------------------------------------- ui previews
def _preview_module():
    # bpy.utils.previews is a lazily imported submodule, not an attribute of bpy.utils.
    from bpy.utils import previews as module

    return module


def preview_icons():
    global _previews
    if _previews is None:
        _previews = _preview_module().new()
    return _previews


def icon_for(path):
    """icon_id for a PNG, loaded once per file version (0 when it cannot be read)."""
    if not path:
        return 0
    path = str(path)
    try:
        stamp = int(Path(path).stat().st_mtime)
    except OSError:
        return 0
    cached = _loaded.get(path)
    if cached and cached[0] == stamp:
        return cached[1]
    icons = preview_icons()
    name = f"is3d-{hashlib.sha1(path.encode('utf-8')).hexdigest()[:16]}"
    if name in icons:
        try:
            icons.remove(name)
        except KeyError:
            pass
    try:
        icons.load(name, path, "IMAGE")
        icon_id = icons[name].icon_id
    except Exception:  # noqa: BLE001
        return 0
    _loaded[path] = (stamp, icon_id)
    return icon_id


def release_previews():
    global _previews
    _loaded.clear()
    if _previews is not None:
        try:
            _preview_module().remove(_previews)
        except Exception:  # noqa: BLE001
            pass
        _previews = None


# --------------------------------------------------------------------- actions
def asset_kind(folder):
    info = _folder_info(Path(folder), _cache_dir(library_root(bpy.context.scene)))
    return info.kind if info is not None else None


def add_to_scene(folder, context, location=None):
    """Append a library asset into the scene; ``location`` defaults to the 3D cursor."""
    folder = Path(folder)
    kind = asset_kind(folder)
    if kind is None:
        raise ValueError("这个目录里没有可用的 IS3D 资产")
    scene = context.scene
    point = tuple(location) if location is not None else tuple(scene.cursor.location)
    return async_ui.append_transfer(folder / TRANSFER_FILE, scene, point, kind)


def reveal(folder):
    path = Path(folder)
    if not path.exists():
        raise FileNotFoundError(str(path))
    try:
        os.startfile(str(path))  # noqa: S606 - Windows shell integration
    except AttributeError:
        bpy.ops.wm.path_open(filepath=str(path))


def delete(folder):
    path = Path(folder).resolve()
    root = Path(library_root(bpy.context.scene)).resolve()
    if root not in path.parents:
        raise ValueError("只能删除资产库目录里的任务")
    shutil.rmtree(path)
    invalidate()
