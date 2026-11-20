"""Asynchronous IS3D image-to-asset tools for Blender."""

bl_info = {
    "name": "IS3D",
    "version": (0, 4, 2),
    "blender": (5, 3, 0),
    "location": "View3D > Sidebar > IS3D",
    "description": "Idol Splat 3D Generator: image to Gaussian splat, textured high and low "
                   "poly meshes; local NVIDIA driver only",
    "category": "Object",
}

import logging
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty

from . import async_ui, library
from .pipeline import BAKE_RESOLUTIONS, HIGH_METHODS, LOW_METHODS

LIBRARY_ROWS = 12


def _format_size(megabytes):
    if megabytes < 0.1:
        return f"{megabytes * 1024:.0f} KB"
    return f"{megabytes:.1f} MB"


def _tag_redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _build_reference_thumb():
    scene = bpy.context.scene
    if scene is not None:
        try:
            library.ensure_reference_thumbnail(scene)
        except Exception:  # noqa: BLE001 - a missing thumbnail must not break the UI
            logging.getLogger(__name__).debug("reference thumbnail failed", exc_info=True)
    _tag_redraw()
    return None


def _schedule_reference_thumb():
    if not bpy.app.timers.is_registered(_build_reference_thumb):
        bpy.app.timers.register(_build_reference_thumb, first_interval=0.05, persistent=False)


def _reference_updated(_self, _context):
    _schedule_reference_thumb()


class IS3D_PG_mesh_options(bpy.types.PropertyGroup):
    reference_image: StringProperty(name="参考图", subtype="FILE_PATH", update=_reference_updated)
    remove_background: BoolProperty(
        name="抠除背景",
        description="用内置 BiRefNet 抠出主体再重建；关闭后按画面中心裁切",
        default=True,
    )
    output_kind: EnumProperty(
        name="产出",
        items=(
            ("SPLAT", "高斯点云", "原生 Gaussian Splat"),
            ("HIGH", "高模", "带颜色属性的重建网格"),
            ("LOW", "低模", "自动展开 UV 并烘焙基础色和法线"),
        ),
        default="LOW",
    )
    high_method: EnumProperty(name="高模算法", items=HIGH_METHODS, default="AUTO")
    low_method: EnumProperty(name="低模算法", items=LOW_METHODS, default="AUTO")
    voxel_size: FloatProperty(name="体素尺寸", default=0.006, min=0.001, max=1.0, precision=4)
    triangles: IntProperty(name="目标三角面数", default=20000, min=256, max=200000)
    resolution: EnumProperty(
        name="贴图分辨率",
        items=[(str(n), f"{n} × {n}", "基础色和法线使用相同的方形分辨率") for n in BAKE_RESOLUTIONS],
        default="2048",
    )
    asset_name: StringProperty(
        name="资产名称",
        description="给这次生成的资产起名；留空则按编号命名（is3d-0001、is3d-0002……）",
    )
    output_directory: StringProperty(name="输出目录", subtype="DIR_PATH")
    show_advanced: BoolProperty(name="高级设置", default=False)
    show_library: BoolProperty(name="资产库", default=True)
    image_mode: EnumProperty(
        name="图像类型",
        items=(("synthetic", "渲染或生成图", "紧凑裁切"), ("real", "实拍照片", "保留更多裁切余量")),
        default="synthetic",
    )
    steps: IntProperty(name="采样步数", default=50, min=1, max=100)
    seed: IntProperty(name="种子", default=42, min=0)
    gaussian_count: EnumProperty(
        name="高斯点数",
        items=[(str(n), f"{n:,}", "生成高斯点数") for n in (32768, 65536, 131072, 262144, 524288, 1048576)],
        default="262144",
    )
    job_progress: FloatProperty(default=0.0, min=0.0, max=1.0, options={"SKIP_SAVE"})
    job_message: StringProperty(options={"SKIP_SAVE"})
    last_job_dir: StringProperty(subtype="DIR_PATH", options={"SKIP_SAVE"})
    job_state: EnumProperty(
        items=[(value, value, "") for value in ("IDLE", "RUNNING", "FINISHED", "CANCELLED", "ERROR")],
        default="IDLE",
        options={"SKIP_SAVE"},
    )


# ------------------------------------------------------------------- operators
class IS3D_OT_build_textured_mesh(bpy.types.Operator):
    bl_idname = "is3d.build_textured_mesh"
    bl_label = "开始生成"
    bl_description = "后台生成并将产物原点放到当前场景的 3D 游标"

    @classmethod
    def poll(cls, context):
        options = context.scene.is3d_mesh_options if context.scene else None
        return options is not None and bool(options.reference_image) and not async_ui.is_busy()

    def execute(self, context):
        try:
            async_ui.start_generation(context.scene)
        except Exception as error:
            logging.getLogger(__name__).exception("Could not start IS3D asset generation")
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class IS3D_OT_cancel_generation(bpy.types.Operator):
    bl_idname = "is3d.cancel_generation"
    bl_label = "取消任务"

    @classmethod
    def poll(cls, _context):
        return async_ui.is_busy()

    def execute(self, _context):
        async_ui.request_cancel()
        return {"FINISHED"}


class IS3D_OT_pick_reference(bpy.types.Operator):
    bl_idname = "is3d.pick_reference"
    bl_label = "选择参考图"
    bl_description = "选择一张参考图"

    filepath: StringProperty(subtype="FILE_PATH")
    filter_image: BoolProperty(default=True, options={"HIDDEN", "SKIP_SAVE"})
    filter_folder: BoolProperty(default=True, options={"HIDDEN", "SKIP_SAVE"})
    filter_glob: StringProperty(default="*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff", options={"HIDDEN"})

    def invoke(self, context, _event):
        if self.filepath:
            return self.execute(context)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if not self.filepath:
            return {"CANCELLED"}
        context.scene.is3d_mesh_options.reference_image = self.filepath
        _schedule_reference_thumb()
        return {"FINISHED"}


class IS3D_OT_refresh_library(bpy.types.Operator):
    bl_idname = "is3d.refresh_library"
    bl_label = "刷新资产库"
    bl_description = "重新扫描输出目录里的历史资产"

    def execute(self, _context):
        library.invalidate()
        _tag_redraw()
        return {"FINISHED"}


class IS3D_OT_add_asset(bpy.types.Operator):
    bl_idname = "is3d.add_asset"
    bl_label = "放入场景"
    bl_description = "把这个资产追加到当前场景的 3D 游标"
    bl_options = {"REGISTER", "UNDO"}

    folder: StringProperty(options={"SKIP_SAVE", "HIDDEN"})

    def execute(self, context):
        try:
            obj = library.add_to_scene(self.folder, context)
        except Exception as error:  # noqa: BLE001 - report instead of raising in the UI
            self.report({"ERROR"}, f"导入失败 · {error}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"已添加 {obj.name}")
        return {"FINISHED"}


class IS3D_OT_drag_asset(bpy.types.Operator):
    bl_idname = "is3d.drag_asset"
    bl_label = "拖入视图"
    bl_description = "按住这个手柄拖到 3D 视图，松开就放到鼠标位置"
    bl_options = {"REGISTER", "UNDO"}

    folder: StringProperty(options={"SKIP_SAVE", "HIDDEN"})

    def invoke(self, context, event):
        if event.type != "LEFTMOUSE" or event.value != "PRESS":
            return {"CANCELLED"}
        self._window = context.window
        self._cursor_set = False
        context.window.cursor_set("PAINT_BRUSH")
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            self._restore_cursor(context)
            location = self._drop_location(context, event.mouse_x, event.mouse_y)
            try:
                obj = library.add_to_scene(self.folder, context, location)
            except Exception as error:  # noqa: BLE001
                self.report({"ERROR"}, f"拖入失败 · {error}")
                return {"CANCELLED"}
            self.report({"INFO"}, f"已放置 {obj.name}")
            return {"FINISHED"}
        if event.type in {"ESC", "RIGHTMOUSE"}:
            self._restore_cursor(context)
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _restore_cursor(self, context):
        if not self._cursor_set:
            self._cursor_set = True
        try:
            context.window.cursor_set("DEFAULT")
        except Exception:  # noqa: BLE001
            pass

    def _drop_location(self, context, x, y):
        window = getattr(self, "_window", None) or context.window
        screen = getattr(window, "screen", None)
        if screen is not None:
            for area in screen.areas:
                if area.type != "VIEW_3D":
                    continue
                if not (area.x <= x < area.x + area.width and area.y <= y < area.y + area.height):
                    continue
                region = next((item for item in area.regions if item.type == "WINDOW"), None)
                space = area.spaces.active
                if region is None or space is None or space.type != "VIEW_3D":
                    continue
                try:
                    from bpy_extras import view3d_utils

                    depth = context.scene.cursor.location.copy()
                    point = view3d_utils.region_2d_to_location_3d(
                        region, space.region_3d, (x - region.x, y - region.y), depth)
                    if point is not None:
                        return point
                except Exception:  # noqa: BLE001
                    break
        return context.scene.cursor.location.copy()


class IS3D_OT_reveal_asset(bpy.types.Operator):
    bl_idname = "is3d.reveal_asset"
    bl_label = "打开目录"
    bl_description = "在文件资源管理器中打开这个任务目录"

    folder: StringProperty(options={"SKIP_SAVE", "HIDDEN"})

    def execute(self, _context):
        try:
            library.reveal(self.folder)
        except Exception as error:  # noqa: BLE001
            self.report({"ERROR"}, f"打不开目录 · {error}")
            return {"CANCELLED"}
        return {"FINISHED"}


class IS3D_OT_delete_asset(bpy.types.Operator):
    bl_idname = "is3d.delete_asset"
    bl_label = "删除资产"
    bl_description = "从磁盘删除这个任务目录"
    bl_options = {"REGISTER", "UNDO"}

    folder: StringProperty(options={"SKIP_SAVE", "HIDDEN"})

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event, message="删除这个资产目录？此操作不可撤销。")

    def execute(self, _context):
        try:
            library.delete(self.folder)
        except Exception as error:  # noqa: BLE001
            self.report({"ERROR"}, f"删除失败 · {error}")
            return {"CANCELLED"}
        _tag_redraw()
        return {"FINISHED"}


# ------------------------------------------------------------------------ panel
class IS3D_PT_mesh_pipeline(bpy.types.Panel):
    bl_idname = "IS3D_PT_mesh_pipeline"
    bl_label = "IS3D 资产生成"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "IS3D"

    # -- reference image card -------------------------------------------------
    def _draw_reference(self, layout, context, options, busy):
        box = layout.box()
        row = box.row()
        thumb = library.reference_thumbnail_path(context.scene)
        icon = library.icon_for(thumb) if thumb and thumb.is_file() else 0
        preview = row.column()
        preview.alignment = "CENTER"
        if icon:
            preview.template_icon(icon_value=icon, scale=5.0)
        else:
            cell = preview.column()
            cell.scale_y = 3.0
            cell.label(text="", icon="IMAGE_REFERENCE")
        info = library.reference_info(context.scene) if options.reference_image else None
        column = row.column()
        if options.reference_image and info:
            column.label(text=Path(bpy.path.abspath(options.reference_image)).name[:24] or "参考图")
            column.label(text=f"{info[0]} × {info[1]}")
            column.label(text=_format_size(info[2]))
        else:
            column.label(text="未选择参考图")
            column.label(text="建议正面、干净的背景")
        buttons = box.row(align=True)
        buttons.enabled = not busy
        buttons.operator(IS3D_OT_pick_reference.bl_idname,
                         text="更换参考图" if options.reference_image else "选择参考图",
                         icon="FILEBROWSER")
        toggle = box.row()
        toggle.enabled = not busy
        toggle.prop(options, "remove_background", icon="MOD_MASK")

    # -- generation settings --------------------------------------------------
    def _draw_settings(self, layout, options, busy):
        column = layout.column()
        column.use_property_split = True
        column.use_property_decorate = False
        column.enabled = not busy
        column.prop(options, "output_kind")
        if options.output_kind != "SPLAT":
            column.prop(options, "high_method")
            column.prop(options, "voxel_size")
        if options.output_kind == "LOW":
            column.prop(options, "low_method")
            column.prop(options, "triangles")
            column.prop(options, "resolution")
        column.prop(options, "asset_name")
        if not options.asset_name.strip():
            hint = column.row()
            hint.enabled = False
            hint.label(text="留空按编号命名，如 is3d-0001", icon="INFO")
        column.prop(options, "output_directory")
        row = column.row()
        row.prop(options, "show_advanced", toggle=True, icon="PREFERENCES")
        if options.show_advanced:
            advanced = layout.column()
            advanced.use_property_split = True
            advanced.use_property_decorate = False
            advanced.enabled = not busy
            advanced.prop(options, "image_mode")
            advanced.prop(options, "reference_image", text="参考图路径")
            advanced.prop(options, "steps")
            advanced.prop(options, "seed")
            advanced.prop(options, "gaussian_count")

    # -- progress and actions -------------------------------------------------
    def _draw_status(self, layout, scene, busy):
        status = async_ui.displayed_status(scene)
        if status.job_state in {"FINISHED", "RUNNING"}:
            percent = int(status.job_progress * 100)
            layout.progress(factor=status.job_progress, type="BAR", text=f"{percent}%")
        if status.job_message and (status.job_state != "RUNNING" or busy):
            icon = {"ERROR": "ERROR", "FINISHED": "CHECKMARK"}.get(status.job_state, "INFO")
            layout.label(text=status.job_message[:85], icon=icon)
        if busy:
            layout.operator(IS3D_OT_cancel_generation.bl_idname, icon="CANCEL")
        else:
            layout.operator(IS3D_OT_build_textured_mesh.bl_idname, icon="PLAY")

    # -- generated asset library ---------------------------------------------
    def _draw_library(self, layout, context, options):
        box = layout.box()
        header = box.row(align=True)
        header.prop(options, "show_library", emboss=False,
                    icon="TRIA_DOWN" if options.show_library else "TRIA_RIGHT", text="资产库")
        header.label(text=library.summary(context.scene))
        header.operator(IS3D_OT_refresh_library.bl_idname, text="", icon="FILE_REFRESH")
        if not options.show_library:
            return
        items = library.scan(context.scene)
        if not items:
            box.label(text="还没有生成过的资产", icon="INFO")
            box.label(text="生成一次后就能在这里复用")
            return
        column = box.column(align=True)
        for item in items[:LIBRARY_ROWS]:
            row = column.row(align=True)
            icon = library.icon_for(item.thumb)
            if icon:
                row.template_icon(icon_value=icon, scale=3.0)
            else:
                placeholder = row.column()
                placeholder.scale_y = 2.6
                placeholder.label(text="", icon=item.icon)
            text = row.column()
            text.label(text=item.title)
            text.label(text=item.detail)
            buttons = row.column(align=True)
            action = buttons.row(align=True)
            action.operator(IS3D_OT_drag_asset.bl_idname, text="", icon="HAND").folder = item.folder
            action.operator(IS3D_OT_add_asset.bl_idname, text="", icon="ADD").folder = item.folder
            manage = buttons.row(align=True)
            manage.operator(IS3D_OT_reveal_asset.bl_idname, text="", icon="FILE_FOLDER").folder = item.folder
            manage.operator(IS3D_OT_delete_asset.bl_idname, text="", icon="TRASH").folder = item.folder
            column.separator(factor=0.4)
        if len(items) > LIBRARY_ROWS:
            box.label(text=f"还有 {len(items) - LIBRARY_ROWS} 个，展开目录查看", icon="THREE_DOTS")
        box.label(text="按住手形按钮拖到视图，或点加号插到游标", icon="INFO")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        options = context.scene.is3d_mesh_options
        busy = async_ui.is_busy()
        self._draw_reference(layout, context, options, busy)
        self._draw_settings(layout, options, busy)
        self._draw_status(layout, context.scene, busy)
        self._draw_library(layout, context, options)


CLASSES = (
    IS3D_PG_mesh_options,
    IS3D_OT_build_textured_mesh,
    IS3D_OT_cancel_generation,
    IS3D_OT_pick_reference,
    IS3D_OT_refresh_library,
    IS3D_OT_add_asset,
    IS3D_OT_drag_asset,
    IS3D_OT_reveal_asset,
    IS3D_OT_delete_asset,
    IS3D_PT_mesh_pipeline,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.is3d_mesh_options = PointerProperty(type=IS3D_PG_mesh_options)
    async_ui.register_handlers()
    _schedule_reference_thumb()


def unregister():
    library.release_previews()
    async_ui.unregister_handlers()
    del bpy.types.Scene.is3d_mesh_options
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
