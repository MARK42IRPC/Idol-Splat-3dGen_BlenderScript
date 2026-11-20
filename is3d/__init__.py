"""Asynchronous IS3D image-to-asset tools for Blender."""

bl_info = {
    "name": "IS3D",
    "version": (0, 4, 0),
    "blender": (5, 3, 0),
    "location": "View3D > Sidebar > IS3D",
    "description": "Idol Splat 3D Generator: image to Gaussian splat, textured high and low "
                   "poly meshes; local NVIDIA driver only",
    "category": "Object",
}

import logging

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty

from . import async_ui
from .pipeline import BAKE_RESOLUTIONS, HIGH_METHODS, LOW_METHODS


class IS3D_PG_mesh_options(bpy.types.PropertyGroup):
    reference_image: StringProperty(name="参考图", subtype="FILE_PATH")
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
    output_directory: StringProperty(name="输出目录", subtype="DIR_PATH")
    show_advanced: BoolProperty(name="高级设置", default=False)
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


class IS3D_PT_mesh_pipeline(bpy.types.Panel):
    bl_idname = "IS3D_PT_mesh_pipeline"
    bl_label = "IS3D 资产生成"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "IS3D"

    def draw(self, context):
        layout = self.layout
        options = context.scene.is3d_mesh_options
        inputs = layout.column()
        inputs.use_property_split = True
        inputs.enabled = not async_ui.is_busy()
        inputs.prop(options, "reference_image")
        inputs.prop(options, "remove_background")
        inputs.prop(options, "output_kind")
        if options.output_kind != "SPLAT":
            inputs.prop(options, "high_method")
            inputs.prop(options, "voxel_size")
        if options.output_kind == "LOW":
            inputs.prop(options, "low_method")
            inputs.prop(options, "triangles")
            baking = inputs.box()
            baking.label(text="烘焙配置")
            baking.prop(options, "resolution")
        inputs.prop(options, "output_directory")
        inputs.prop(options, "show_advanced", toggle=True)
        if options.show_advanced:
            inputs.prop(options, "image_mode")
            inputs.prop(options, "steps")
            inputs.prop(options, "seed")
            inputs.prop(options, "gaussian_count")

        status = async_ui.displayed_status(context.scene)
        if status.job_state == "FINISHED" or (status.job_state == "RUNNING" and async_ui.is_busy()):
            percent = int(status.job_progress * 100)
            layout.progress(factor=status.job_progress, type="BAR", text=f"{percent}%")
        if status.job_message and (status.job_state != "RUNNING" or async_ui.is_busy()):
            icon = {"ERROR": "ERROR", "FINISHED": "CHECKMARK"}.get(status.job_state, "INFO")
            layout.label(text=status.job_message[:85], icon=icon)
        if async_ui.is_busy():
            layout.operator(IS3D_OT_cancel_generation.bl_idname, icon="CANCEL")
        else:
            layout.operator(IS3D_OT_build_textured_mesh.bl_idname, icon="PLAY")


CLASSES = (IS3D_PG_mesh_options, IS3D_OT_build_textured_mesh, IS3D_OT_cancel_generation, IS3D_PT_mesh_pipeline)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.is3d_mesh_options = PointerProperty(type=IS3D_PG_mesh_options)
    async_ui.register_handlers()


def unregister():
    async_ui.unregister_handlers()
    del bpy.types.Scene.is3d_mesh_options
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
