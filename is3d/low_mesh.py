"""Build a triangulated bake target from an evaluated high-resolution mesh."""

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree


QUADRIFLOW_MIN_TRIANGLES = 32


def _mesh_arrays(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        points = np.empty((len(mesh.vertices), 3), dtype=np.float32)
        triangles = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
        mesh.vertices.foreach_get("co", points.ravel())
        mesh.loop_triangles.foreach_get("vertices", triangles.ravel())
        transform = np.asarray(evaluated.matrix_world, dtype=np.float64)
        points = points @ transform[:3, :3].T + transform[:3, 3]
        if np.linalg.det(transform[:3, :3]) < 0:
            triangles = triangles[:, [0, 2, 1]]
        return points, triangles
    finally:
        evaluated.to_mesh_clear()


def _parts(points, triangles):
    """Group vertex-connected triangles, with compact vertex IDs per part."""
    parent = np.arange(len(points), dtype=np.int32)
    while True:
        roots = parent[triangles]
        if np.all(roots == roots[:, :1]):
            break
        np.minimum.at(parent, roots.ravel(), np.repeat(roots.min(axis=1), 3))
        while np.any(parent[parent] != parent):
            parent = parent[parent]

    labels = parent[triangles[:, 0]]
    order = np.argsort(labels, kind="stable")
    breaks = np.flatnonzero(np.diff(labels[order])) + 1
    parts = []
    for indices in np.split(order, breaks):
        vertices, inverse = np.unique(triangles[indices], return_inverse=True)
        parts.append((points[vertices], inverse.reshape(-1, 3)))
    return parts


def _budgets(parts, target, minimum=4):
    face_counts = np.array([len(tris) for _, tris in parts])
    quotas = np.minimum(face_counts, minimum)
    weighted_area = []
    for points, triangles in parts:
        a, b, c = points[triangles].transpose(1, 0, 2)
        weighted_area.append(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
    weighted_area = np.maximum(weighted_area, np.finfo(float).tiny)
    left = min(int(face_counts.sum()), target) - int(quotas.sum())
    while left > 0:
        available = np.flatnonzero(quotas < face_counts)
        weights = weighted_area[available]
        shares = left * (weights / weights.sum())
        room = face_counts[available] - quotas[available]
        extra = np.minimum(shares.astype(int), room)
        if not extra.any():
            extra[np.argsort(-shares, kind="stable")[:left]] = 1
        quotas[available] += extra
        left -= int(extra.sum())
    return quotas


def _new_object(name, points, triangles, collection):
    mesh = bpy.data.meshes.new(name)
    obj = None
    try:
        mesh.from_pydata(points.tolist(), [], triangles.tolist(), shade_flat=False)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        collection.objects.link(obj)
        return obj
    except Exception:
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
        raise


def _remesh_part(points, triangles, limit, method, collection, depsgraph):
    if method == "QEM" and len(triangles) <= limit:
        return points, triangles
    if method == "QUADRIFLOW" and len(triangles) < 4:
        raise RuntimeError("QuadriFlow requires at least four faces per component.")

    working_points = points
    if method == "QUADRIFLOW":
        # Blender's manifold check rejects edges within a fixed 1e-4 in local
        # space, even on a consistently oriented mesh. Fit the temporary copy
        # in a scale where valid short edges clear that numerical cutoff.
        center = (points.max(axis=0) + points.min(axis=0)) * 0.5
        corners = points[triangles]
        edge_size = float(np.abs(corners - corners[:, [1, 2, 0]]).max(axis=-1).min())
        if edge_size <= 0.0:
            raise RuntimeError("QuadriFlow requires nonzero-length edges.")
        solver_scale = max(1.0, 2e-4 / edge_size)
        working_points = (points - center) * solver_scale
        if np.abs(working_points).max() > 1e6:
            raise RuntimeError("QuadriFlow input has edges too short relative to the component size.")

    obj = _new_object("IS3D low temporary", working_points, triangles, collection)
    source_mesh = obj.data
    try:
        if method == "QEM":
            modifier = obj.modifiers.new("Quadric collapse", "DECIMATE")
            modifier.decimate_type = "COLLAPSE"
            modifier.ratio = limit / len(triangles)
            modifier.use_collapse_triangulate = True
        else:
            layer_objects = bpy.context.view_layer.objects
            previous = layer_objects.active
            try:
                layer_objects.active = obj
                with bpy.context.temp_override(object=obj, active_object=obj):
                    result = bpy.ops.object.quadriflow_remesh(
                        mode="FACES",
                        target_faces=max(QUADRIFLOW_MIN_TRIANGLES, limit) // 2,
                        use_mesh_symmetry=False,
                        use_preserve_boundary=True,
                        seed=0,
                    )
                if "FINISHED" not in result:
                    raise RuntimeError(
                        "QuadriFlow could not remesh this component. It requires an oriented manifold surface."
                    )
            finally:
                layer_objects.active = previous
        bpy.context.view_layer.update()
        result_points, result_triangles = _mesh_arrays(obj, depsgraph)
        if not len(result_triangles):
            raise RuntimeError(f"{method} produced an empty component.")
        if method == "QUADRIFLOW":
            result_points = result_points / solver_scale + center
        return result_points, result_triangles
    finally:
        result_mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in (source_mesh,) if result_mesh == source_mesh else (source_mesh, result_mesh):
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)


def _face_normals(points, triangles):
    a, b, c = points[triangles].transpose(1, 0, 2)
    return np.cross(b - a, c - a)


def _join_parts(parts):
    points = []
    triangles = []
    offset = 0
    for part_points, part_triangles in parts:
        points.append(part_points)
        triangles.append(part_triangles + offset)
        offset += len(part_points)
    return np.concatenate(points), np.concatenate(triangles)


def _clean_quadriflow(points, triangles):
    face_area = np.linalg.norm(_face_normals(points, triangles), axis=1)
    valid = np.isfinite(face_area) & (face_area > 0)
    if not valid.any():
        raise RuntimeError("QuadriFlow produced no nondegenerate faces.")
    parts = _parts(points, triangles[valid])
    total_area = float(face_area[valid].sum())
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    retained = []
    for vertices, faces in parts:
        area = np.linalg.norm(_face_normals(vertices, faces), axis=1).sum()
        span = np.linalg.norm(np.ptp(vertices, axis=0))
        if len(faces) <= 12 and area < total_area * 1e-4 and span < diagonal * 0.01:
            continue
        retained.append((vertices, faces))
    # Always keep the output if all fragments were below the cleanup thresholds.
    retained = retained or parts
    return (*_join_parts(retained), int((~valid).sum()), len(parts) - len(retained))


def _vacuum(points, triangles, shell, faces):
    """Tighten a surface-seeded shell with a decreasing contact offset."""
    surface = BVHTree.FromPolygons(points.tolist(), triangles.tolist(), all_triangles=True)
    sides = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(sides, axis=1), axis=0)
    source, neighbor = np.concatenate([edges, edges[:, ::-1]]).T
    degree = np.maximum(np.bincount(source, minlength=len(shell)), 1)
    edge_size = float(np.median(np.linalg.norm(shell[edges[:, 0]] - shell[edges[:, 1]], axis=1)))
    start_gap = min(edge_size * 0.4, float(np.linalg.norm(np.ptp(points, axis=0))) * 0.02)
    max_slide = edge_size * 0.2

    contacts = np.empty_like(shell)
    normals = np.empty_like(shell)
    for index, vertex in enumerate(shell):
        hit, normal, _, _ = surface.find_nearest(vertex)
        if hit is None:
            raise RuntimeError("The vacuum envelope could not find its source surface.")
        contacts[index] = hit
        normals[index] = normal

    def stable_step(previous, desired):
        old_normals = _face_normals(previous, faces)
        delta = desired - previous
        for fraction in (1.0, 0.5, 0.25, 0.125):
            candidate = previous + delta * fraction
            new_normals = _face_normals(candidate, faces)
            if np.all(np.einsum("ij,ij->i", old_normals, new_normals) > 0.0):
                return candidate, fraction
        return previous, 0.0

    shell, _ = stable_step(shell, contacts + start_gap * normals)
    blocked = 0
    for gap in np.linspace(start_gap, 0.0, 9)[1:]:
        averaged = np.zeros_like(shell)
        np.add.at(averaged, source, shell[neighbor])
        tangent = averaged / degree[:, None] - shell
        tangent -= normals * np.einsum("ij,ij->i", tangent, normals)[:, None]
        length = np.linalg.norm(tangent, axis=1)
        tangent *= np.minimum(0.35, max_slide / np.maximum(length, 1e-30))[:, None]

        projected = contacts.copy()
        projected_normals = normals.copy()
        for index, proposed in enumerate(shell + tangent):
            hit, normal, _, distance = surface.find_nearest(proposed)
            if (
                hit is not None
                and distance <= start_gap + max_slide
                and np.dot(normal, normals[index]) > 0.15
            ):
                projected[index] = hit
                projected_normals[index] = normal
        shell, fraction = stable_step(shell, projected + gap * projected_normals)
        if fraction:
            contacts = projected
            normals = projected_normals
        blocked += int(fraction < 1.0)
    return shell, blocked


def build_low(high, collection, method="AUTO", target_triangles=20000, progress=print):
    """Return an identity-transform, triangulated object and a method report."""
    requested = method.upper()
    selected = "QEM" if requested == "AUTO" else requested
    if selected not in {"QEM", "QUADRIFLOW", "VACUUM"}:
        raise ValueError(f"Unknown low-mesh method {method!r}.")
    if target_triangles < 4:
        raise ValueError("The low-mesh triangle target must be at least four.")
    if high.type != "MESH" or bpy.context.mode != "OBJECT":
        raise ValueError("Building a low mesh requires a mesh source in Object Mode.")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    points, triangles = _mesh_arrays(high, depsgraph)
    if not len(triangles) or not np.isfinite(points).all():
        raise ValueError("The evaluated high mesh needs finite vertices and faces.")
    valid_faces = np.linalg.norm(_face_normals(points, triangles), axis=1) > 0
    triangles = triangles[valid_faces]
    if not len(triangles):
        raise ValueError("The evaluated high mesh has only degenerate triangles.")

    parts = _parts(points, triangles)
    # Very small targets are often infeasible for closed components in
    # QuadriFlow. Reserve a working minimum before allocating the rest by area.
    minimum = QUADRIFLOW_MIN_TRIANGLES if selected == "QUADRIFLOW" else 4
    budgets = _budgets(parts, target_triangles, minimum=minimum)
    progress(f"低模 {selected} · {len(parts):,} 个部件 · 目标 {target_triangles:,} 三角面")
    output_parts = []
    blocked = 0
    removed_degenerate = 0
    removed_fragments = 0
    for index, ((part_points, part_faces), budget) in enumerate(zip(parts, budgets), start=1):
        try:
            base_method = "QEM" if selected == "VACUUM" else selected
            low_points, low_faces = _remesh_part(
                part_points, part_faces, int(budget), base_method, collection, depsgraph
            )
            if selected == "VACUUM":
                low_points, reduced = _vacuum(part_points, part_faces, low_points, low_faces)
                blocked += reduced
            if selected == "QUADRIFLOW":
                low_points, low_faces, degenerate, fragments = _clean_quadriflow(low_points, low_faces)
                removed_degenerate += degenerate
                removed_fragments += fragments
        except RuntimeError as error:
            raise RuntimeError(f"{selected} failed on component {index} of {len(parts)}. {error}") from error
        output_parts.append((low_points, low_faces))

    low_points, low_faces = _join_parts(output_parts)
    warnings = []
    if not valid_faces.all():
        warnings.append(f"Ignored {int((~valid_faces).sum())} degenerate source triangles.")
    low_part_count = len(_parts(low_points, low_faces))
    if len(parts) != low_part_count:
        warnings.append("The low mesh has a different component count from the high mesh.")
    if len(low_faces) > target_triangles:
        warnings.append("The result exceeds the triangle budget. Separate components and topology limit further reduction.")
    if removed_degenerate or removed_fragments:
        warnings.append(
            f"Removed {removed_degenerate} degenerate triangles and {removed_fragments} microscopic QuadriFlow fragments."
        )
    if selected == "VACUUM":
        warnings.append("Geometric envelope experiment. Narrow concavities may bridge between sampled contact points.")
        if blocked:
            warnings.append(f"Reduced or skipped {blocked} contraction steps to avoid face flips.")
    low = _new_object("IS3D Low", low_points, low_faces, collection)
    return low, {
        "method_requested": requested,
        "method": selected,
        "implementation": {
            "QEM": "blender_quadric_decimate",
            "QUADRIFLOW": "blender_quadriflow_then_triangulate",
            "VACUUM": "geometric_contact_envelope",
        }[selected],
        "coordinate_space": "evaluated source world coordinates, identity result transform",
        "target_triangles": target_triangles,
        "high_triangles": len(triangles),
        "low_triangles": len(low_faces),
        "high_components": len(parts),
        "low_components": low_part_count,
        "warnings": warnings,
    }
