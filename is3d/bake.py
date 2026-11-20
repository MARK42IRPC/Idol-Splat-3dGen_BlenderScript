"""Build a UV atlas and transfer an IS3D surface to a low-poly mesh with Cycles."""

from pathlib import Path
import math

import bpy
import numpy as np


def _select(target, *sources):
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    for source in sources:
        source.select_set(True)
    bpy.context.view_layer.objects.active = target


def _uv_coordinates(mesh, layer):
    values = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    layer.uv.foreach_get("vector", values)
    return values.reshape(-1, 2)


def _pack_uv(margin_uv, shape_method):
    bpy.ops.uv.pack_islands(
        udim_source="CLOSEST_UDIM", shape_method=shape_method, rotate=True,
        rotate_method="CARDINAL" if shape_method == "AABB" else "ANY",
        margin_method="FRACTION", margin=margin_uv,
    )


def _uv_mask(mesh, layer, size, offenders=None):
    uv = _uv_coordinates(mesh, layer)
    if not np.isfinite(uv).all() or uv.min() < -1e-6 or uv.max() > 1.000001:
        raise ValueError(f"UV atlas extends outside 0–1 (range {uv.min():.5g} to {uv.max():.5g})")
    mesh.calc_loop_triangles()
    mask = np.zeros((size, size), dtype=bool)
    overlap = np.zeros_like(mask)
    degenerate = 0
    for triangle in mesh.loop_triangles:
        points = uv[list(triangle.loops)].astype(np.float64) * size
        first, last = points.min(axis=0), points.max(axis=0)
        x0, y0 = np.maximum(np.ceil(first - 0.5).astype(int), 0)
        x1, y1 = np.minimum(np.floor(last - 0.5).astype(int) + 1, size)
        a, b, c = points
        ab, ac = b - a, c - a
        det = ab[0] * ac[1] - ab[1] * ac[0]
        if abs(det) < 1e-9:
            degenerate += 1
            continue
        if x0 >= x1 or y0 >= y1:
            continue
        x = np.arange(x0, x1)[None, :] + 0.5 - a[0]
        y = np.arange(y0, y1)[:, None] + 0.5 - a[1]
        u = (x * ac[1] - y * ac[0]) / det
        v = (ab[0] * y - ab[1] * x) / det
        # Exclude shared triangle edges so adjacent faces do not count as overlap.
        inside = (u > 1e-7) & (v > 1e-7) & (u + v < 1.0 - 1e-7)
        region = np.s_[y0:y1, x0:x1]
        collision = mask[region] & inside
        overlap[region] |= collision
        if offenders is not None and collision.any():
            offenders.add(triangle.polygon_index)
        mask[region] |= inside
    return mask, int(np.count_nonzero(overlap)), degenerate


def _chart_count(mesh):
    """Count connected face charts after converting Smart UV's cuts to seams."""
    parents = list(range(len(mesh.polygons)))
    seams = [edge.use_seam for edge in mesh.edges]
    first_face = {}
    charts = len(parents)

    def root(face):
        while parents[face] != face:
            parents[face] = parents[parents[face]]
            face = parents[face]
        return face

    for face in mesh.polygons:
        for loop in face.loop_indices:
            edge = mesh.loops[loop].edge_index
            if seams[edge]:
                continue
            adjacent = first_face.setdefault(edge, face.index)
            first, second = root(face.index), root(adjacent)
            if first != second:
                parents[first] = second
                charts -= 1
    return charts


def _repair_uv_overlap(mesh, layer, original, resolution, padding, offenders):
    """Move a handful of folded faces to an isolated strip and verify the result."""
    border = padding + 1
    span = 8
    pitch = 2 * border + span
    per_column = resolution // pitch
    if not per_column:
        raise ValueError("No room to repair UV overlap")
    repaired = set(offenders)
    for _ in range(3):
        strip = math.ceil(len(repaired) / per_column) * pitch
        if len(repaired) > 128 or strip > 0.08 * resolution:
            break
        scale = (resolution - strip) / resolution
        if border * scale < padding:
            break
        updated = original * scale
        updated[:, 1] += (1.0 - scale) * 0.5
        for tile, face in enumerate(sorted(repaired)):
            col, row = divmod(tile, per_column)
            x = resolution - strip + col * pitch + border
            y = row * pitch + border
            loops = list(mesh.polygons[face].loop_indices)
            updated[loops] = np.asarray(((x, y), (x + span, y), (x, y + span))) / resolution
        layer.uv.foreach_set("vector", updated.ravel())
        mesh.update()
        remaining = set()
        mask, overlap, degenerate = _uv_mask(mesh, layer, resolution, remaining)
        if not overlap and not degenerate and mask.any():
            for face in repaired:
                for loop in mesh.polygons[face].loop_indices:
                    mesh.edges[mesh.loops[loop].edge_index].use_seam = True
            return mask, len(repaired)
        repaired |= remaining
    raise ValueError(f"Could not isolate UV overlap by repacking {len(repaired)} faces")


def _pack_and_validate(low, layer_name, coords, resolution, padding, shape_method="CONCAVE"):
    mesh = low.data
    mesh.uv_layers[layer_name].uv.foreach_set("vector", coords.ravel())
    mesh.update()
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        _pack_uv((padding + 1) / resolution, shape_method)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    layer = mesh.uv_layers[layer_name]
    uv = _uv_coordinates(mesh, layer)
    if np.isfinite(uv).all():
        center = (uv.min(axis=0) + uv.max(axis=0)) * 0.5
        tile = np.floor(center)
        if (tile != 0).any():
            uv -= tile
            layer.uv.foreach_set("vector", uv.ravel())
            mesh.update()
    offenders = set()
    mask, overlap, degenerate = _uv_mask(mesh, layer, resolution, offenders)
    if overlap and not degenerate and 0 < len(offenders) <= 128 and overlap <= max(64, mask.sum() * 0.002):
        return _repair_uv_overlap(mesh, layer, uv, resolution, padding, offenders)
    if overlap or degenerate or not mask.any():
        raise ValueError(f"UV atlas measured {overlap} overlapping texels and {degenerate} degenerate triangles")
    return mask, 0


def _smart_layout(low, layer_name, angle):
    mesh = low.data
    mesh.edges.foreach_set("use_seam", np.zeros(len(mesh.edges), dtype=bool))
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.select_all(action="SELECT")
        result = bpy.ops.uv.smart_project(
            angle_limit=math.radians(angle), island_margin=0.0, correct_aspect=False,
        )
        if result != {"FINISHED"}:
            raise RuntimeError(f"Smart UV returned {result}")
        bpy.ops.uv.seams_from_islands(mark_seams=True, mark_sharp=False)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    return _uv_coordinates(mesh, mesh.uv_layers[layer_name]), _chart_count(mesh)


def _grid_atlas(mesh, layer_name, resolution, margin):
    """Pack two counterclockwise right triangles per cell with explicit edge gutters."""
    charts = len(mesh.polygons)
    grid = math.ceil(math.sqrt((charts + 1) // 2))
    cell = resolution / grid
    effective_margin = min(margin, max(0, int((cell - 2.0) / (2.0 + math.sqrt(2.0))) - 1))
    border = effective_margin + 1
    diagonal = border * math.sqrt(2.0)
    span = cell - 2.0 * border - diagonal
    if span < 1.0:
        raise RuntimeError(f"Texture {resolution} cannot fit {charts} per-face UV charts with two-pixel gaps")

    uv = np.empty((len(mesh.loops), 2), dtype=np.float32)
    for face in mesh.polygons:
        tile, slot = divmod(face.index, 2)
        row, col = divmod(tile, grid)
        left, bottom = col * cell + border, row * cell + border
        right, top = left + cell - 2.0 * border, bottom + cell - 2.0 * border
        if slot == 0:
            coords = ((left, bottom), (left + span, bottom), (left, bottom + span))
        else:
            coords = ((right, top), (right - span, top), (right, top - span))
        uv[list(face.loop_indices)] = coords
    uv /= resolution
    layer = mesh.uv_layers[layer_name]
    layer.uv.foreach_set("vector", uv.ravel())
    mesh.edges.foreach_set("use_seam", np.ones(len(mesh.edges), dtype=bool))
    mesh.update()
    mask, overlap, degenerate = _uv_mask(mesh, layer, resolution)
    if overlap or degenerate or not mask.any():
        raise RuntimeError(f"Per-face atlas failed ({overlap} overlaps and {degenerate} degenerate triangles)")
    return mask, effective_margin, charts


def _unwrap(low, resolution, margin):
    mesh = low.data
    layer_name = "IS3DUV"
    layer = mesh.uv_layers.get(layer_name) or mesh.uv_layers.new(name=layer_name)
    mesh.uv_layers.active = layer
    layer.active_render = True
    _select(low)
    settings = bpy.context.scene.tool_settings
    previous_sync = settings.use_uv_select_sync
    settings.use_uv_select_sync = True
    try:
        _, charts = _smart_layout(low, layer_name, 88)
        # Reserve most of the atlas for geometry, even for thousands of small seams.
        effective_margin = min(margin, max(1, int(0.21 * resolution / math.sqrt(charts)) - 1))

        method = "SLIM"
        fallback = None
        trials = []
        chart_angle = 88
        repaired_faces = 0
        try:
            bpy.ops.object.mode_set(mode="EDIT")
            result = bpy.ops.uv.unwrap(
                method="MINIMUM_STRETCH", margin_method="ADD", margin=0.0,
                iterations=10, no_flip=True, correct_aspect=False,
            )
            bpy.ops.object.mode_set(mode="OBJECT")
            if result != {"FINISHED"}:
                raise RuntimeError(f"SLIM returned {result}")
            slim_uv = _uv_coordinates(mesh, mesh.uv_layers[layer_name])
            mask, repaired_faces = _pack_and_validate(low, layer_name, slim_uv, resolution, effective_margin)
        except (ValueError, RuntimeError) as error:
            fallback = str(error)
            trials.append({"method": "SLIM", "error": fallback})
            method = "Smart UV"
            if low.mode == "EDIT":
                bpy.ops.object.mode_set(mode="OBJECT")
            for angle in (45, 30, 20, 15):
                try:
                    smart_uv, charts = _smart_layout(low, layer_name, angle)
                    effective_margin = min(margin, max(1, int(0.21 * resolution / math.sqrt(charts)) - 1))
                    mask, repaired_faces = _pack_and_validate(low, layer_name, smart_uv, resolution, effective_margin, "AABB")
                    chart_angle = angle
                    break
                except (ValueError, RuntimeError) as smart_error:
                    trials.append({"method": "Smart UV", "angle_degrees": angle, "error": str(smart_error)})
            else:
                method = "Per-face grid"
                chart_angle = None
                mask, effective_margin, charts = _grid_atlas(mesh, layer_name, resolution, margin)
    finally:
        if low.mode == "EDIT":
            bpy.ops.object.mode_set(mode="OBJECT")
        settings.use_uv_select_sync = previous_sync

    count = int(np.count_nonzero(mask))
    return layer_name, mask, effective_margin, {
        "method": method, "slim_fallback_reason": fallback, "atlas_texels": count,
        "atlas_utilization": count / mask.size, "overlap_texels": 0, "degenerate_triangles": 0,
        "charts": _chart_count(mesh) if repaired_faces else charts, "chart_angle_degrees": chart_angle,
        "repacked_faces": repaired_faces, "failed_attempts": trials,
        "requested_border_px": margin + 1, "packing_border_px": effective_margin + 1,
    }


def _new_image(name, resolution, colorspace):
    image = bpy.data.images.new(name, resolution, resolution, alpha=True)
    image.colorspace_settings.name = colorspace
    image.file_format = "PNG"
    return image


def _cycles_bake(pass_type, image_node, image, uv_layer, ray_distance, margin):
    image_node.image = image
    result = bpy.ops.object.bake(
        type=pass_type, target="IMAGE_TEXTURES", save_mode="INTERNAL",
        use_selected_to_active=True, use_clear=True, uv_layer=uv_layer,
        use_cage=False, max_ray_distance=2.0 * ray_distance, cage_extrusion=ray_distance,
        margin=margin, margin_type="EXTEND", normal_space="TANGENT",
        normal_r="POS_X", normal_g="POS_Y", normal_b="POS_Z",
    )
    if result != {"FINISHED"}:
        raise RuntimeError(f"Cycles {pass_type} bake returned {result}")


def _hit_count(image, uv_mask):
    pixels = np.empty(uv_mask.size * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return int(np.count_nonzero(uv_mask.ravel() & (pixels[3::4] > 0.5)))


def _source_material():
    material = bpy.data.materials.new("IS3D Color Source")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    color = nodes.new("ShaderNodeVertexColor")
    color.layer_name = "IS3DColor"
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(color.outputs["Color"], emit.inputs["Color"])
    material.node_tree.links.new(emit.outputs["Emission"], output.inputs["Surface"])
    return material


def unwrap_and_bake(high, low, output_dir, resolution=2048, progress=print, *, margin=16, min_coverage=0.98):
    """Create one nonoverlapping UV tile and an sRGB appearance plus a +Y tangent normal map."""
    if not 0 < min_coverage <= 1 or not 1 <= margin < resolution // 8:
        raise ValueError("Choose valid coverage and a bake margin below one eighth of the resolution")
    if high == low or high.type != "MESH" or low.type != "MESH" or not len(low.data.polygons):
        raise ValueError("Expected distinct nonempty high and low meshes")
    if any(len(face.vertices) != 3 for face in low.data.polygons):
        raise ValueError("Triangulate the low mesh before unwrapping and baking")
    color = high.data.color_attributes.get("IS3DColor")
    if color is None or color.data_type != "FLOAT_COLOR":
        raise ValueError("High mesh must supply scene-linear FLOAT_COLOR 'IS3DColor'")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress("Unwrapping and packing one UV tile")
    uv_layer, uv_mask, bake_margin, uv_report = _unwrap(low, resolution, margin)
    uv_count = uv_report["atlas_texels"]

    source = _source_material()
    source_slots = list(high.data.materials)
    source_indices = np.empty(len(high.data.polygons), dtype=np.int32)
    high.data.polygons.foreach_get("material_index", source_indices)
    baked = bpy.data.materials.new("IS3D Baked")
    baked.use_nodes = True
    low.data.materials.clear()
    low.data.materials.append(baked)
    nodes = baked.node_tree.nodes
    for node in nodes:
        node.select = False
    color_node = nodes.new("ShaderNodeTexImage")
    color_node.label = "IS3D Appearance"
    color_node.location = (-560, 240)
    color_node.select = True
    nodes.active = color_node
    base_color = _new_image("IS3D Base Color", resolution, "sRGB")
    normal_map = _new_image("IS3D Tangent Normal", resolution, "Non-Color")

    scene = bpy.context.scene
    previous_engine = scene.render.engine
    previous_device = scene.cycles.device
    previous_samples = scene.cycles.samples
    ray_trials = []
    try:
        high.data.materials.clear()
        high.data.materials.append(source)
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 1
        _select(low, high)
        bpy.context.view_layer.update()
        diagonal = low.dimensions.length
        if diagonal <= 0:
            raise ValueError("Low mesh has no usable bounds")
        for fraction in (0.01, 0.025, 0.05):
            ray_distance = diagonal * fraction
            _cycles_bake("EMIT", color_node, base_color, uv_layer, ray_distance, margin=0)
            count = _hit_count(base_color, uv_mask)
            coverage = count / uv_count
            progress(f"Cycles ray distance {ray_distance:.5g}: {coverage:.2%} UV coverage")
            ray_trials.append({
                "start_offset": ray_distance, "max_ray_distance": ray_distance * 2.0,
                "hit_texels": count, "coverage": coverage,
            })
            if coverage >= min_coverage:
                break
        else:
            trials = ", ".join(f"{item['coverage']:.2%}" for item in ray_trials)
            raise RuntimeError(f"Cycles projection coverage {trials}; required {min_coverage:.2%}")

        progress("Baking padded sRGB appearance and tangent-space normal maps")
        _cycles_bake("EMIT", color_node, base_color, uv_layer, ray_distance, bake_margin)
        _cycles_bake("NORMAL", color_node, normal_map, uv_layer, ray_distance, bake_margin)
    finally:
        scene.cycles.samples = previous_samples
        scene.cycles.device = previous_device
        scene.render.engine = previous_engine
        high.data.materials.clear()
        for material in source_slots:
            high.data.materials.append(material)
        if source_slots:
            high.data.polygons.foreach_set("material_index", source_indices)
        bpy.data.materials.remove(source)
        _select(low)

    normal_path = output / "normal.png"
    color_path = output / "base_color.png"
    base_color.filepath_raw = str(color_path)
    normal_map.filepath_raw = str(normal_path)
    base_color.save()
    normal_map.save()
    color_node.image = base_color
    normal_node = nodes.new("ShaderNodeTexImage")
    normal_node.image = normal_map
    normal_node.label = "IS3D Tangent Normal"
    normal_node.location = (-560, -80)
    uv_node = nodes.new("ShaderNodeUVMap")
    uv_node.uv_map = uv_layer
    uv_node.location = (-800, 240)
    normal = nodes.new("ShaderNodeNormalMap")
    normal.space = "TANGENT"
    normal.convention = "OPENGL"
    normal.uv_map = uv_layer
    normal.location = (-280, -60)
    principled = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.65
    links = baked.node_tree.links
    links.new(uv_node.outputs["UV"], color_node.inputs["Vector"])
    links.new(uv_node.outputs["UV"], normal_node.inputs["Vector"])
    links.new(color_node.outputs["Color"], principled.inputs["Base Color"])
    links.new(normal_node.outputs["Color"], normal.inputs["Color"])
    links.new(normal.outputs["Normal"], principled.inputs["Normal"])
    return {
        "resolution": resolution, "requested_margin_px": margin, "margin_px": bake_margin, "uv": uv_report,
        "ray_trials": ray_trials, "minimum_hit_coverage": min_coverage,
        "hit_coverage": coverage, "device": "CPU", "normal_convention": "OpenGL +Y",
        "base_color": color_path.name, "normal": normal_path.name,
    }
