"""Build a polygonal surface from a native Blender Gaussian point cloud."""

import math

import bpy
import numpy as np


MIN_OPACITY = 0.02
MAX_GRID_VERTICES = 18_000_000
MAX_FIELD_SAMPLES = 150_000_000
MAX_SDF_BOX_VOXELS = 24_000_000
SDF_BAND_WIDTH = 3


def read_gaussians(source):
    """Return world positions, local scales, local WXYZ rotations, and SH0/opacity."""
    if source.type != 'POINTCLOUD' or source.data.type != 'GAUSSIAN_SPLAT':
        raise ValueError('The source must be a native Gaussian Splat point cloud.')
    count = len(source.data.points)
    if not count:
        raise ValueError('The Gaussian Splat point cloud is empty.')

    def read(name, field, width):
        attribute = source.data.attributes.get(name)
        if attribute is None or len(attribute.data) != count:
            raise ValueError(f'The Gaussian Splat is missing point attribute {name!r}.')
        values = np.empty((count, width), dtype=np.float32)
        attribute.data.foreach_get(field, values.ravel())
        return values

    transform = np.asarray(source.matrix_world, dtype=np.float32)
    positions = read('position', 'vector', 3) @ transform[:3, :3].T + transform[:3, 3]
    return (positions, read('scale', 'vector', 3), read('rotation', 'value', 4),
            read('radiance:base', 'vector', 4))


def _support_mask(positions, alpha, voxel_size):
    """Drop visually faint points in cells with little nearby support."""
    cells = np.floor((positions - positions.min(axis=0)) / (2.0 * voxel_size)).astype(np.int64)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    sparse = counts[inverse] < 3
    if not sparse.any():
        return np.ones(len(positions), dtype=bool)
    # Adjacent cells may carry a thin feature that crosses a cell boundary.
    stride_y, stride_z = (cells.max(axis=0) + 3)[1:]
    strides = np.array([stride_y * stride_z, stride_z, 1], dtype=np.int64)
    keys = (cells + 1) @ strides
    cell_keys, occupancy = np.unique(keys, return_counts=True)
    candidates = np.flatnonzero(sparse & (alpha < 0.12))
    nearby = np.zeros(len(candidates), dtype=np.int32)
    for delta in np.ndindex(3, 3, 3):
        query = keys[candidates] + (np.asarray(delta) - 1) @ strides
        index = np.searchsorted(cell_keys, query).clip(max=len(cell_keys) - 1)
        nearby += np.where(cell_keys[index] == query, occupancy[index], 0).astype(np.int32)
    keep = np.ones(len(positions), dtype=bool)
    keep[candidates] = nearby >= 3
    return keep


def _gaussian_basis(scales, quaternions, transform):
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    w, x, y, z = quaternions.T
    basis = np.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ], axis=1).reshape(-1, 3, 3)
    return (transform @ basis) * scales[:, None, :]


def _voxel_mesh(positions, basis, voxel_size, collection):
    # The spherical fog fallback uses the shorter axes to limit broad needles and haze.
    axes = np.sort(np.linalg.norm(basis, axis=1), axis=1)
    radii = (2.2 * np.sqrt(axes[:, 0] * axes[:, 1])).clip(1.65 * voxel_size, 2.8 * voxel_size)
    points = bpy.data.meshes.new('IS3D reconstruction samples')
    working = bpy.data.objects.new('IS3D reconstruction samples', points)
    nodes = bpy.data.node_groups.new('IS3D Points to Mesh', 'GeometryNodeTree')
    try:
        collection.objects.link(working)
        points.vertices.add(len(positions))
        points.vertices.foreach_set('co', np.ascontiguousarray(positions).ravel())
        attribute = points.attributes.new('is3d_radius', 'FLOAT', 'POINT')
        attribute.data.foreach_set('value', np.ascontiguousarray(radii, dtype=np.float32))
        points.update()

        nodes.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        nodes.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        group_input = nodes.nodes.new('NodeGroupInput')
        group_output = nodes.nodes.new('NodeGroupOutput')
        radii_field = nodes.nodes.new('GeometryNodeInputNamedAttribute')
        radii_field.data_type = 'FLOAT'
        radii_field.inputs['Name'].default_value = 'is3d_radius'
        as_points = nodes.nodes.new('GeometryNodeMeshToPoints')
        as_points.mode = 'VERTICES'
        volume = nodes.nodes.new('GeometryNodePointsToVolume')
        volume.inputs['Resolution Mode'].default_value = 'Size'
        volume.inputs['Voxel Size'].default_value = voxel_size
        surface = nodes.nodes.new('GeometryNodeVolumeToMesh')
        surface.inputs['Threshold'].default_value = 0.1
        surface.inputs['Adaptivity'].default_value = 0.10
        nodes.links.new(group_input.outputs['Geometry'], as_points.inputs['Mesh'])
        nodes.links.new(as_points.outputs['Points'], volume.inputs['Points'])
        nodes.links.new(radii_field.outputs['Attribute'], volume.inputs['Radius'])
        nodes.links.new(volume.outputs['Volume'], surface.inputs['Volume'])
        nodes.links.new(surface.outputs['Mesh'], group_output.inputs['Geometry'])
        modifier = working.modifiers.new('Reconstruct surface', 'NODES')
        modifier.node_group = nodes
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        return bpy.data.meshes.new_from_object(working.evaluated_get(depsgraph), depsgraph=depsgraph)
    finally:
        bpy.data.objects.remove(working, do_unlink=True)
        bpy.data.node_groups.remove(nodes)
        bpy.data.meshes.remove(points)


def _splat_field(positions, basis, alpha, voxel_size):
    covariance = basis @ basis.swapaxes(1, 2)
    variances = np.diagonal(covariance, axis1=1, axis2=2)
    determinant = np.linalg.det(covariance).clip(min=0)
    identity = np.eye(3, dtype=np.float32)
    step = voxel_size
    for _ in range(12):
        prefilter = (0.48 * step) ** 2
        radii = 3.0 * np.sqrt(variances + prefilter)
        origin = (positions - radii).min(axis=0) - 2.0 * step
        bounds = (positions + radii).max(axis=0) + 2.0 * step
        grid_shape = np.ceil((bounds - origin) / step).astype(np.int64) + 1
        lower = np.ceil((positions - radii - origin) / step).astype(np.int32)
        upper = np.floor((positions + radii - origin) / step).astype(np.int32)
        widths = upper - lower + 1
        grid_count = math.prod(grid_shape.tolist())
        footprints = np.prod(widths, axis=1, dtype=np.int64)
        sample_count = footprints.sum()
        ratio = max(grid_count / MAX_GRID_VERTICES, sample_count / MAX_FIELD_SAMPLES,
                    footprints.max() / 350_000)
        if ratio <= 1:
            break
        step *= max(1.15, float(ratio) ** (1 / 3) * 1.05)
    else:
        raise RuntimeError('The ellipsoid surface exceeded its bounded grid workspace.')

    # Band-limit sub-voxel detail without increasing a splat's integrated mass.
    filtered = covariance + prefilter * identity
    weight = alpha * np.sqrt(determinant / np.linalg.det(filtered))
    inverse = np.linalg.inv(filtered)
    field = np.zeros(grid_count, dtype=np.float32)
    _, shape_ids = np.unique(widths, axis=0, return_inverse=True)
    order = np.argsort(shape_ids)
    splits = np.flatnonzero(np.diff(shape_ids[order])) + 1
    strides = np.array([grid_shape[1] * grid_shape[2], grid_shape[2], 1], dtype=np.int64)

    for group in np.split(order, splits):
        shape = tuple(widths[group[0]])
        offsets = np.moveaxis(np.indices(shape, dtype=np.int32), 0, -1).reshape(-1, 3)
        batch = max(1, 180_000 // len(offsets))
        for start in range(0, len(group), batch):
            selected = group[start:start + batch]
            indices = lower[selected, None, :] + offsets[None]
            delta = indices.astype(np.float32) * step + origin - positions[selected, None, :]
            exponent = np.einsum('nki,nij,nkj->nk', delta, inverse[selected], delta)
            inside = exponent <= 9.0
            values = weight[selected, None] * np.exp(-0.5 * exponent)
            np.add.at(field, indices[inside] @ strides, values[inside])
    return field.reshape(tuple(grid_shape)), origin, step, int(sample_count)


def _iso_level(field):
    occupied = field[field > field.max() * 1e-3]
    if not occupied.size:
        raise RuntimeError('No occupied voxels were generated from the Gaussian Splat.')
    upper = float(np.quantile(occupied, 0.995))
    hist, bounds = np.histogram(np.minimum(occupied, upper), bins=128, range=(0, upper))
    centers = (bounds[:-1] + bounds[1:]) * 0.5
    left = np.cumsum(hist)
    right = hist.sum() - left
    moments = np.cumsum(hist * centers)
    between = (moments[-1] * left - moments * hist.sum()) ** 2 / np.maximum(left * right, 1)
    return float(0.4 * centers[np.argmax(between)])


def _tetrahedron_cases():
    """Oriented crossing edges for a positively oriented tetrahedron."""
    reference = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cases = {}
    for mask in range(1, 15):
        inner = [index for index in range(4) if mask & (1 << index)]
        outer = [index for index in range(4) if index not in inner]
        if len(inner) == 2:
            edges = [(inner[0], outer[0]), (inner[0], outer[1]),
                     (inner[1], outer[1]), (inner[1], outer[0])]
        else:
            edges = [(a, b) for a in inner for b in outer]
        mids = reference[edges].mean(axis=1)
        outward = reference[outer].mean(axis=0) - reference[inner].mean(axis=0)
        if np.dot(np.cross(mids[1] - mids[0], mids[2] - mids[0]), outward) < 0:
            edges.reverse()
        cases[mask] = np.asarray([edges[:3]] if len(edges) == 3
                                 else [edges[:3], [edges[0], edges[2], edges[3]]])
    return cases


def _marching_tetrahedra(field, origin, step, level):
    inside = field > level
    cell_shape = np.asarray(field.shape) - 1
    offsets = np.array(list(np.ndindex(2, 2, 2)), dtype=np.int64)
    mask = np.zeros(tuple(cell_shape), dtype=np.uint8)
    for index, offset in enumerate(offsets):
        sl = tuple(slice(start, start + length) for start, length in zip(offset, cell_shape))
        mask |= inside[sl].astype(np.uint8) << index
    active = (mask != 0) & (mask != 255)
    states = mask[active]
    if not len(states):
        raise RuntimeError('The anisotropic density field has no surface at its iso-level.')
    if len(states) > 700_000:
        raise RuntimeError('The ellipsoid surface has too many active cells. Increase voxel size.')

    cells = np.ravel_multi_index(np.nonzero(active), field.shape)
    strides = np.array([field.shape[1] * field.shape[2], field.shape[2], 1])
    corner_offsets = offsets @ strides
    tetrahedra = [(0, 4, 6, 7), (0, 6, 2, 7), (0, 2, 3, 7),
                  (0, 3, 1, 7), (0, 1, 5, 7), (0, 5, 4, 7)]
    cases = _tetrahedron_cases()
    face_keys = []
    for tetra in map(np.array, tetrahedra):
        masks = ((states[:, None] >> tetra) & 1) @ np.array([1, 2, 4, 8])
        for pattern, crossing in cases.items():
            selected = cells[masks == pattern]
            if not len(selected):
                continue
            grid_edges = np.sort(tetra[crossing], axis=-1)
            start, end = grid_edges[..., 0], grid_edges[..., 1]
            edge_start = selected[:, None, None] + corner_offsets[start][None]
            # A grid edge has a unique lower endpoint and one of seven offsets.
            face_keys.append((edge_start * 8 + (start ^ end)).reshape(-1, 3))

    edge_keys, faces = np.unique(np.concatenate(face_keys), return_inverse=True)
    edge_start, direction = np.divmod(edge_keys, 8)
    vectors = offsets[direction]
    edge_end = edge_start + vectors @ strides
    first, last = field.ravel()[edge_start], field.ravel()[edge_end]
    ratio = ((level - first) / (last - first)).clip(2e-5, 1 - 2e-5)
    coords = np.column_stack(np.unravel_index(edge_start, field.shape))
    vertices = (coords + ratio[:, None] * vectors) * step + origin
    return vertices, faces.reshape(-1, 3).astype(np.int32)


def _ellipsoid_mesh(positions, basis, alpha, voxel_size):
    field, origin, step, samples = _splat_field(positions, basis, alpha, voxel_size)
    level = _iso_level(field)
    vertices, triangles = _marching_tetrahedra(field, origin, step, level)
    mesh = _triangle_mesh('IS3D Ellipsoid surface', vertices, triangles)
    return mesh, {'iso_level': level, 'grid_shape': list(field.shape), 'field_samples': samples}, step


def _triangle_mesh(name, vertices, triangles):
    mesh = bpy.data.meshes.new(name)
    try:
        mesh.vertices.add(len(vertices))
        mesh.vertices.foreach_set('co', np.ascontiguousarray(vertices, dtype=np.float32).ravel())
        mesh.loops.add(triangles.size)
        mesh.loops.foreach_set('vertex_index', triangles.ravel())
        mesh.polygons.add(len(triangles))
        mesh.polygons.foreach_set('loop_start', np.arange(0, triangles.size, 3, dtype=np.int32))
        mesh.update(calc_edges=True)
    except Exception:
        bpy.data.meshes.remove(mesh)
        raise
    return mesh


def _components(edges, vertex_count):
    roots = np.arange(vertex_count, dtype=np.int32)
    while len(edges):
        left, right = roots[edges].T
        different = left != right
        if not different.any():
            break
        np.minimum.at(roots, np.maximum(left[different], right[different]),
                      np.minimum(left[different], right[different]))
        while True:
            parents = roots[roots]
            if np.array_equal(parents, roots):
                break
            roots = parents
    return np.unique(roots, return_inverse=True, return_counts=True)[1:]


def _mesh_positions(mesh):
    positions = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get('co', positions.ravel())
    return positions


def _mesh_edges(mesh):
    edges = np.empty((len(mesh.edges), 2), dtype=np.int32)
    mesh.edges.foreach_get('vertices', edges.ravel())
    return edges


def _edge_uses(mesh):
    used = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get('edge_index', used)
    return np.bincount(used, minlength=len(mesh.edges))


def _mesh_triangles(mesh):
    mesh.calc_loop_triangles()
    triangles = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
    mesh.loop_triangles.foreach_get('vertices', triangles.ravel())
    return triangles


def _component_surfaces(mesh, positions, labels, centers):
    triangles = _mesh_triangles(mesh)
    area = np.zeros(len(centers), dtype=np.float64)
    volume = np.zeros(len(centers), dtype=np.float64)
    for start in range(0, len(triangles), 250_000):
        faces = triangles[start:start + 250_000]
        component = labels[faces[:, 0]]
        # Measure relative to nearby origins for stability in translated scenes.
        first, second, third = (positions[faces] - centers[component, None]).transpose(1, 0, 2)
        normals = np.cross(second - first, third - first)
        area += np.bincount(component, 0.5 * np.linalg.norm(normals, axis=1), minlength=len(area))
        volume += np.bincount(component, np.einsum('ij,ij->i', first, normals) / 6, minlength=len(area))
    return area, volume


def _taubin_smooth(mesh, positions, edges, iterations):
    first, second = edges.T
    deg = np.bincount(edges.ravel(), minlength=len(positions)).clip(min=1)
    for _ in range(iterations):
        for factor in (0.20, -0.21):
            neighbors = np.zeros_like(positions)
            np.add.at(neighbors, first, positions[second])
            np.add.at(neighbors, second, positions[first])
            positions += factor * (neighbors / deg[:, None] - positions)
    mesh.vertices.foreach_set('co', positions.ravel())
    mesh.update()


def _clean_mesh(mesh, voxel_size, smoothing):
    if not len(mesh.polygons):
        raise RuntimeError('High mesh reconstruction returned no faces.')
    mesh.validate(clean_customdata=False)
    edges = _mesh_edges(mesh)
    positions = _mesh_positions(mesh)
    labels, counts = _components(edges, len(positions))
    lower = np.full((len(counts), 3), np.inf, dtype=np.float32)
    upper = np.full((len(counts), 3), -np.inf, dtype=np.float32)
    np.minimum.at(lower, labels, positions)
    np.maximum.at(upper, labels, positions)
    area, signed_volume = _component_surfaces(mesh, positions, labels, (lower + upper) * 0.5)
    extent = (upper - lower).max(axis=1)
    open_edges = edges[_edge_uses(mesh) != 2]
    closed = np.ones(len(counts), dtype=bool)
    closed[labels[open_edges[:, 0]]] = False
    inner = closed & (signed_volume < 0)
    # A detached component must have enough visible area or enough length to be a thin feature.
    keep = (counts >= 16) & ((area >= 112.0 * voxel_size**2) | (extent >= 10.0 * voxel_size)) & ~inner
    keep[area.argmax()] = True
    if not keep.all():
        import bmesh

        editable = bmesh.new()
        try:
            editable.from_mesh(mesh)
            discarded = [vertex for vertex in editable.verts if not keep[labels[vertex.index]]]
            bmesh.ops.delete(editable, geom=discarded, context='VERTS')
            editable.to_mesh(mesh)
            mesh.update()
        finally:
            editable.free()

        positions = _mesh_positions(mesh)
        edges = _mesh_edges(mesh)
    if smoothing:
        _taubin_smooth(mesh, positions, edges, smoothing)

    return {
        **_mesh_report(mesh, positions, int(keep.sum())),
        'removed_components': int((~keep).sum()),
        'removed_inner_components': int((inner & ~keep).sum()),
        'smoothing_iterations': smoothing,
    }


def _mesh_report(mesh, positions, component_count):
    edge_use = _edge_uses(mesh)
    mesh.polygons.foreach_set('use_smooth', np.ones(len(mesh.polygons), dtype=bool))
    return {
        'vertices': len(mesh.vertices),
        'faces': len(mesh.polygons),
        'triangles': len(mesh.loops) - 2 * len(mesh.polygons),
        'components': component_count,
        'euler_characteristic': len(mesh.vertices) - len(mesh.edges) + len(mesh.polygons),
        'boundary_edges': int((edge_use == 1).sum()),
        'non_manifold_edges': int(((edge_use == 0) | (edge_use > 2)).sum()),
        'world_bounds': [positions.min(axis=0).tolist(), positions.max(axis=0).tolist()],
    }


def _sdf_grid_mesh(seed, collection, voxel_size, smooth):
    working = bpy.data.objects.new('IS3D SDF work surface', seed)
    group = bpy.data.node_groups.new('IS3D Signed Distance Refit', 'GeometryNodeTree')
    try:
        collection.objects.link(working)
        group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        group_input = group.nodes.new('NodeGroupInput')
        group_output = group.nodes.new('NodeGroupOutput')
        grid = group.nodes.new('GeometryNodeMeshToSDFGrid')
        grid.inputs['Voxel Size'].default_value = voxel_size
        grid.inputs['Band Width'].default_value = SDF_BAND_WIDTH
        group.links.new(group_input.outputs['Geometry'], grid.inputs['Mesh'])
        distance = grid.outputs['SDF Grid']
        if smooth:
            mean = group.nodes.new('GeometryNodeSDFGridMean')
            mean.inputs['Width'].default_value = 1
            mean.inputs['Iterations'].default_value = 1
            group.links.new(distance, mean.inputs['Grid'])
            distance = mean.outputs['Grid']
        surface = group.nodes.new('GeometryNodeGridToMesh')
        surface.inputs['Threshold'].default_value = 0.0
        surface.inputs['Adaptivity'].default_value = 0.10
        group.links.new(distance, surface.inputs['Grid'])
        group.links.new(surface.outputs['Mesh'], group_output.inputs['Geometry'])
        modifier = working.modifiers.new('Refit signed distance surface', 'NODES')
        modifier.node_group = group
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        return bpy.data.meshes.new_from_object(working.evaluated_get(depsgraph), depsgraph=depsgraph)
    finally:
        bpy.data.objects.remove(working, do_unlink=True)
        bpy.data.node_groups.remove(group)


def _fit_sdf_component(points, triangles, collection, grid_voxel, seed_voxel):
    from mathutils.bvhtree import BVHTree

    seed = _triangle_mesh('IS3D SDF seed component', points, triangles)
    tree = BVHTree.FromPolygons(points.tolist(), triangles.tolist(), all_triangles=True)
    seed_euler = len(seed.vertices) - len(seed.edges) + len(seed.polygons)
    bounds = np.stack((points.min(axis=0), points.max(axis=0)))
    rejections = []
    try:
        for smooth in (True, False):
            candidate = _sdf_grid_mesh(seed, collection, grid_voxel, smooth)
            try:
                try:
                    stats = _clean_mesh(candidate, grid_voxel, smoothing=0)
                except RuntimeError as error:
                    rejections.append(str(error))
                    continue
                if stats['components'] != 1 or stats['boundary_edges'] or stats['non_manifold_edges']:
                    rejections.append('refit changed the closed connected surface')
                    continue
                if seed_euler >= -14 and stats['euler_characteristic'] != seed_euler:
                    rejections.append('refit changed the Euler characteristic of a low-genus component')
                    continue
                vertices = _mesh_positions(candidate)
                bbox_shift = float(np.abs(np.asarray(stats['world_bounds']) - bounds).max())
                samples = vertices[np.random.default_rng(42).choice(len(vertices), min(2048, len(vertices)), replace=False)]
                distances = np.array([tree.find_nearest(point)[3] for point in samples])
                displacement = float(np.quantile(distances, 0.99))
                if bbox_shift > seed_voxel or displacement > seed_voxel:
                    rejections.append('refit moved its bounds or sampled surface by more than one seed voxel')
                    continue
                return vertices, _mesh_triangles(candidate), {
                    'sdf_applied': True, 'smoothed': smooth, 'seed_euler': seed_euler,
                    'euler': stats['euler_characteristic'], 'bounds_shift': bbox_shift,
                    'refit_to_seed_vertex_distance_p99': displacement, 'rejected_attempts': rejections,
                }
            finally:
                bpy.data.meshes.remove(candidate)
    finally:
        bpy.data.meshes.remove(seed)
    return points, triangles, {
        'sdf_applied': False, 'smoothed': False, 'seed_euler': seed_euler, 'euler': seed_euler,
        'bounds_shift': 0.0, 'refit_to_seed_vertex_distance_p99': 0.0, 'rejected_attempts': rejections,
    }


def _edge_roughness(mesh):
    """Compare length-weighted normal jumps after triangulating both surfaces."""
    points = _mesh_positions(mesh)
    faces = _mesh_triangles(mesh)
    a, b, c = points[faces].transpose(1, 0, 2)
    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1)
    first, second = faces.ravel(), faces[:, [1, 2, 0]].ravel()
    keys = np.minimum(first, second).astype(np.int64) * len(points) + np.maximum(first, second)
    order = np.argsort(keys)
    ordered = keys[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1]
    pairs = starts[np.diff(np.r_[starts, len(order)]) == 2]
    face_a, face_b = order[pairs] // 3, order[pairs + 1] // 3
    valid = (norms[face_a] > 0) & (norms[face_b] > 0)
    face_a, face_b = face_a[valid], face_b[valid]
    cosine = np.einsum('ni,ni->n', normals[face_a], normals[face_b]) / (norms[face_a] * norms[face_b])
    degrees = np.degrees(np.arccos(cosine.clip(-1.0, 1.0)))
    edges = order[pairs[valid]]
    lengths = np.linalg.norm(points[first[edges]] - points[second[edges]], axis=1)
    return {
        'length_weighted_mean_dihedral_degrees': float(np.average(degrees, weights=lengths)),
        'edge_length_fraction_over_45_degrees': float(lengths[degrees > 45.0].sum() / lengths.sum()),
    }


def _refit_signed_distance(seed, collection, voxel_size, seed_report, progress):
    seed_points = _mesh_positions(seed)
    seed_faces = _mesh_triangles(seed)
    labels, counts = _components(_mesh_edges(seed), len(seed_points))
    if seed_report['boundary_edges'] or seed_report['non_manifold_edges']:
        raise RuntimeError('SDF refit requires a closed manifold seed. Try a larger seed voxel size.')
    extent = np.ptp(seed_points, axis=0)
    grid_voxel = voxel_size * 0.5
    while True:
        shape = tuple((np.ceil(extent / grid_voxel) + 2 * SDF_BAND_WIDTH + 2).astype(int))
        ratio = math.prod(shape) / MAX_SDF_BOX_VOXELS
        if ratio <= 1:
            break
        grid_voxel *= max(1.10, ratio ** (1 / 3) * 1.05)

    # Refit islands independently so the grid cannot fuse nearby disconnected parts.
    face_labels = labels[seed_faces[:, 0]]
    order = np.argsort(face_labels, kind='stable')
    splits = np.flatnonzero(np.diff(face_labels[order])) + 1
    vertices = []
    triangles = []
    details = []
    offset = 0
    for group in np.split(order, splits):
        indices, local = np.unique(seed_faces[group], return_inverse=True)
        points, faces, detail = _fit_sdf_component(seed_points[indices], local.reshape(-1, 3), collection,
                                                   grid_voxel, voxel_size)
        vertices.append(points)
        triangles.append(faces + offset)
        details.append(detail)
        offset += len(points)
    applied = sum(row['sdf_applied'] for row in details)
    if not applied:
        raise RuntimeError('SDF refit could not preserve any component within the geometry guards. '
                           'Try a smaller seed voxel size or select the VOXEL method.')
    points = np.concatenate(vertices)
    mesh = _triangle_mesh('IS3D signed distance surface', points, np.concatenate(triangles))
    try:
        statistics = _mesh_report(mesh, points, len(counts))
        euler_changes = sum(row['seed_euler'] != row['euler'] for row in details)
        warnings = []
        if euler_changes:
            warnings.append('Euler characteristic changed on a high-genus component. Small cavities or tunnels may differ.')
        if applied != len(counts):
            warnings.append('Components that did not pass the SDF geometry guards retained their seed surfaces.')
        report = {
            'representation': 'OpenVDB mesh-to-level-set narrow band, zero level extraction',
            'sdf_applied': True,
            'sdf_smoothing_applied': any(row['smoothed'] for row in details),
            'sdf_voxel_size': grid_voxel,
            'sdf_band_width_voxels': SDF_BAND_WIDTH,
            'sdf_dense_bbox_shape_bound': [int(side) for side in shape],
            'sdf_mean_filter_width_voxels': 1,
            'sdf_mean_filter_iterations': 1,
            'sdf_exact_euler_guard_max_seed_genus': 8,
            'sdf_components_refitted': applied,
            'sdf_components_kept_from_seed': len(counts) - applied,
            'sdf_components_with_euler_changes': euler_changes,
            'sdf_component_results': details,
            'sdf_warnings': warnings,
            'seed_triangles': seed_report['triangles'],
            'seed_euler_characteristic': seed_report['euler_characteristic'],
            'seed_world_bounds': seed_report['world_bounds'],
            'seed_roughness': _edge_roughness(seed),
            'sdf_roughness': _edge_roughness(mesh),
        }
    except Exception:
        bpy.data.meshes.remove(mesh)
        raise
    progress(f'SDF refitted {applied} of {len(counts)} components at voxel size {grid_voxel:.6g}.')
    return mesh, report, statistics


def build_high(source, collection, method='AUTO', voxel_size=0.006, progress=print):
    """Return an identity-transform high mesh and its reconstruction statistics."""
    requested = method.upper()
    method = 'VOXEL' if requested == 'AUTO' else requested
    if method not in {'VOXEL', 'ELLIPSOID', 'SDF'}:
        raise ValueError(f'Unknown high-mesh method {requested!r}.')
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError('Voxel size must be a finite positive number.')

    positions, scales, rotations, radiance = read_gaussians(source)
    matrix = np.asarray(source.matrix_world, dtype=np.float32)[:3, :3]
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('The Gaussian Splat must have an invertible world transform.')
    alpha = radiance[:, 3]
    valid = np.isfinite(positions).all(axis=1) & np.isfinite(scales).all(axis=1)
    valid &= np.isfinite(rotations).all(axis=1) & np.isfinite(alpha)
    valid &= (scales > 0).all(axis=1) & (np.linalg.norm(rotations, axis=1) > 1e-8)
    visible = valid & (alpha >= MIN_OPACITY) & (alpha <= 1.0)
    if not visible.any():
        raise ValueError('No valid splats met the minimum opacity threshold.')
    keep = np.flatnonzero(visible)
    keep = keep[_support_mask(positions[keep], alpha[keep], voxel_size)]
    if not len(keep):
        raise ValueError('No splats remained after opacity and spatial support filtering.')
    progress(f'{method} high mesh uses {len(keep):,} of {len(positions):,} Gaussian splats.')
    basis = _gaussian_basis(scales[keep], rotations[keep], matrix)

    detail = {}
    effective = voxel_size
    stats = None
    if method == 'VOXEL':
        mesh = _voxel_mesh(positions[keep], basis, effective, collection)
    elif method == 'SDF':
        seed = _voxel_mesh(positions[keep], basis, effective, collection)
        try:
            seed_report = _clean_mesh(seed, effective, smoothing=2)
            mesh, detail, stats = _refit_signed_distance(seed, collection, effective, seed_report, progress)
        finally:
            bpy.data.meshes.remove(seed)
    else:
        mesh, detail, effective = _ellipsoid_mesh(positions[keep], basis, alpha[keep], effective)
    try:
        if stats is None:
            stats = _clean_mesh(mesh, effective, smoothing=2 if method == 'VOXEL' else 0)
        mesh.name = 'IS3D High Surface'
        result = bpy.data.objects.new('IS3D High Surface', mesh)
        try:
            collection.objects.link(result)
        except Exception:
            bpy.data.objects.remove(result, do_unlink=True)
            raise
    except Exception:
        bpy.data.meshes.remove(mesh)
        raise

    report = {
        'requested_method': requested,
        'method': method,
        'requested_voxel_size': voxel_size,
        'voxel_size': effective,
        'input_splats': len(positions),
        'retained_splats': len(keep),
        'minimum_opacity': MIN_OPACITY,
        'invalid_splats': int((~valid).sum()),
        'opacity_removed_splats': int(valid.sum() - visible.sum()),
        'sparse_removed_splats': int(visible.sum() - len(keep)),
        **detail,
        **stats,
    }
    progress(f'High mesh contains {stats["triangles"]:,} triangles and {stats["components"]:,} components.')
    return result, report
