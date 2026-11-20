"""Measure Blender IS3D mesh reconstruction and simplification without inference or baking."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "validation" / "results" / "baseline" / "asset.ply"
LOW_METHODS = ("QEM", "QUADRIFLOW", "VACUUM")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snowman-ply", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--voxel-size", type=float, default=0.006)
    parser.add_argument("--snowman-budget", type=int, default=20000)
    parser.add_argument("--torus-budget", type=int, default=1200)
    parser.add_argument("--samples", type=int, default=8192, help="Surface samples per direction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-methods", nargs="+", choices=LOW_METHODS, default=LOW_METHODS)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if (not np.isfinite(args.voxel_size) or args.voxel_size <= 0 or args.samples < 100 or args.seed < 0
            or min(args.snowman_budget, args.torus_budget) < 32):
        parser.error("Choose positive finite voxel size, at least 100 samples, a nonnegative seed, and budgets of at least 32")
    return args


def import_snowman(path):
    before = set(bpy.context.scene.objects)
    status = bpy.ops.wm.ply_import(
        filepath=str(path), forward_axis="Z", up_axis="NEGATIVE_Y", global_scale=1.0,
        use_scene_unit=False, merge_verts=False, import_attributes=True,
    )
    imported = set(bpy.context.scene.objects) - before
    if status != {"FINISHED"} or len(imported) != 1:
        raise ValueError(f"Could not import exactly one snowman object from {path}")
    source = imported.pop()
    if source.type != "POINTCLOUD" or source.data.type != "GAUSSIAN_SPLAT":
        raise ValueError(f"Expected an IS3D Gaussian Splat in {path}")
    bpy.context.view_layer.update()
    return source


def _torus(major, minor, center, rings, segments):
    u = np.arange(rings)[:, None] * (2 * np.pi / rings)
    v = np.arange(segments)[None, :] * (2 * np.pi / segments)
    rho = major + minor * np.cos(v)
    z = np.broadcast_to(minor * np.sin(v), (rings, segments))
    points = np.stack((rho * np.cos(u), rho * np.sin(u), z), axis=-1).reshape(-1, 3) + center
    i, j = np.indices((rings, segments))
    a = (i * segments + j).ravel()
    b = (((i + 1) % rings) * segments + j).ravel()
    c = (((i + 1) % rings) * segments + (j + 1) % segments).ravel()
    d = (i * segments + (j + 1) % segments).ravel()
    return points, np.concatenate((np.stack((a, b, c), axis=1), np.stack((a, c, d), axis=1)))


def make_tori(collection):
    large, large_faces = _torus(0.30, 0.095, (0.0, 0.0, 0.0), rings=96, segments=40)
    small, small_faces = _torus(0.110, 0.032, (0.63, 0.12, 0.09), rings=64, segments=32)
    points = np.concatenate((large, small))
    triangles = np.concatenate((large_faces, small_faces + len(large)))
    mesh = bpy.data.meshes.new("Benchmark two disjoint tori")
    mesh.from_pydata(points.tolist(), [], triangles.tolist(), shade_flat=False)
    mesh.update()
    obj = bpy.data.objects.new(mesh.name, mesh)
    collection.objects.link(obj)
    bpy.context.view_layer.update()
    return obj


class Surface:
    def __init__(self, obj):
        from is3d.low_mesh import _mesh_arrays

        self.points, self.triangles = _mesh_arrays(obj, bpy.context.evaluated_depsgraph_get())
        if not len(self.triangles) or not np.isfinite(self.points).all():
            raise ValueError("Expected a non-empty finite triangulated surface")
        sides = self.points[self.triangles]
        self.area = np.linalg.norm(np.cross(sides[:, 1] - sides[:, 0], sides[:, 2] - sides[:, 0]), axis=1) * 0.5
        self.diagonal = float(np.linalg.norm(np.ptp(self.points, axis=0)))
        self._bvh = None

    @property
    def bvh(self):
        if self._bvh is None:
            good = self.triangles[self.area > 0]
            if not len(good):
                raise ValueError("No nondegenerate triangles to query")
            self._bvh = BVHTree.FromPolygons(self.points.tolist(), good.tolist(), all_triangles=True)
        return self._bvh

    def sample(self, count, seed):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.triangles), count, p=self.area / self.area.sum())
        corners = self.points[self.triangles[indices]]
        radial = np.sqrt(rng.random(count))
        theta = rng.random(count)
        weights = np.stack((1 - radial, radial * (1 - theta), radial * theta), axis=1)
        return np.einsum("ni,nij->nj", weights, corners)

    def topology(self):
        from is3d.high_mesh import _components

        first = self.triangles.ravel()
        second = self.triangles[:, [1, 2, 0]].ravel()
        vertices = len(self.points)
        key = np.minimum(first, second).astype(np.int64) * vertices + np.maximum(first, second)
        keys, reverse, incidence = np.unique(key, return_inverse=True, return_counts=True)
        edges = np.column_stack(np.divmod(keys, vertices)).astype(np.int32)
        _, component_counts = _components(edges, vertices)
        orientation_balance = np.bincount(reverse, np.where(first < second, 1, -1))
        return {
            "vertices": vertices,
            "triangles": len(self.triangles),
            "edges": len(edges),
            "components": len(component_counts),
            "euler_characteristic": vertices - len(edges) + len(self.triangles),
            "boundary_edges": int((incidence == 1).sum()),
            "nonmanifold_edges": int((incidence > 2).sum()),
            "orientation_conflicts": int(((incidence == 2) & (orientation_balance != 0)).sum()),
            "degenerate_triangles": int((self.area <= 1e-15 * self.diagonal**2).sum()),
            "bbox_diagonal": self.diagonal,
        }


def distances(samples, target):
    tree = target.bvh
    values = np.empty(len(samples), dtype=np.float64)
    for index, point in enumerate(samples):
        _, _, _, values[index] = tree.find_nearest(point)
    return values


def summarize(values, diagonal):
    p50, p95, p99 = np.percentile(values, (50, 95, 99))
    return {
        "p50": float(p50), "p95": float(p95), "p99": float(p99),
        "p95_bbox_ratio": float(p95 / diagonal),
        "sample_max": float(values.max()),
    }


def compare(reference, result, count, seed, reference_samples=None):
    if reference_samples is None:
        reference_samples = reference.sample(count, seed)
    forward = summarize(distances(reference_samples, result), reference.diagonal)
    backward = summarize(distances(result.sample(count, seed + 1), reference), reference.diagonal)
    return {"reference_to_result": forward, "result_to_reference": backward,
            "symmetric_p95_bbox_ratio": max(forward["p95_bbox_ratio"], backward["p95_bbox_ratio"])}


def benchmark_low(obj, surface, collection, budget, args):
    from is3d.low_mesh import build_low

    samples = surface.sample(args.samples, args.seed)
    results = {}
    for method in dict.fromkeys(args.low_methods):
        print(f"Low {obj.name} · {method} · {budget:,} triangles", flush=True)
        start = perf_counter()
        result = None
        try:
            result, report = build_low(obj, collection, method=method, target_triangles=budget)
            runtime = perf_counter() - start
            candidate = Surface(result)
            results[method] = {
                "status": "passed",
                "build_seconds": round(runtime, 3),
                "build": report,
                "topology": candidate.topology(),
                "fit": compare(surface, candidate, args.samples, args.seed, samples),
            }
        except Exception as error:
            print(f"{method} failed · {error}", flush=True)
            results[method] = {"status": "failed", "elapsed_seconds": round(perf_counter() - start, 3),
                               "error": str(error)}
        finally:
            if result is not None:
                mesh = result.data
                bpy.data.objects.remove(result, do_unlink=True)
                bpy.data.meshes.remove(mesh)
    return {"target_triangles": budget, "reference_topology": surface.topology(), "methods": results}


def main():
    args = arguments()
    if not bpy.app.background or bpy.app.version < (5, 3, 0) or bpy.data.filepath:
        raise RuntimeError("Start Blender 5.3+ in background mode without opening a blend file")
    sys.path.insert(0, str(ROOT))
    from is3d.high_mesh import build_high
    from is3d.pipeline import HIGH_METHODS

    path = args.snowman_ply.resolve(strict=True)
    with path.open("rb") as fixture:
        checksum = hashlib.file_digest(fixture, "sha256").hexdigest()
    source = import_snowman(path)
    collection = bpy.data.collections.new("IS3D Benchmark Geometry")
    bpy.context.scene.collection.children.link(collection)
    high_rows = {}
    high_objects = {}
    high_surfaces = {}
    for method, _, _ in HIGH_METHODS:
        if method == "AUTO":
            continue
        print(f"High snowman · {method}", flush=True)
        start = perf_counter()
        try:
            obj, report = build_high(source, collection, method, args.voxel_size)
            elapsed = perf_counter() - start
            surface = Surface(obj)
            high_rows[method] = {"status": "passed", "build_seconds": round(elapsed, 3),
                                 "build": report, "topology": surface.topology()}
            high_objects[method], high_surfaces[method] = obj, surface
        except Exception as error:
            print(f"{method} failed · {error}", flush=True)
            high_rows[method] = {"status": "failed", "elapsed_seconds": round(perf_counter() - start, 3),
                                 "error": str(error)}

    snowman_low = None
    high_disagreement = None
    if "VOXEL" in high_surfaces:
        reference = high_surfaces["VOXEL"]
        for method, surface in high_surfaces.items():
            if method != "VOXEL":
                high_rows[method]["fit_to_voxel"] = compare(reference, surface, args.samples, args.seed)
        high_disagreement = high_rows.get("ELLIPSOID", {}).get("fit_to_voxel")
        snowman_low = benchmark_low(high_objects["VOXEL"], reference, collection, args.snowman_budget, args)
    tori = make_tori(collection)
    torus_low = benchmark_low(tori, Surface(tori), collection, args.torus_budget, args)

    compared = {
        "blender": bpy.app.version_string,
        "blender_build_hash": bpy.app.build_hash.decode(),
        "numpy": np.__version__,
        "seed": args.seed,
        "samples_per_direction": args.samples,
        "voxel_size": args.voxel_size,
        "snowman_ply": str(path),
        "snowman_ply_sha256": checksum,
        "measurement": "Area-weighted samples to nearest triangle, normalized by reference bounding-box diagonal",
        "limitations": [
            "The snowman has no ground-truth mesh. The high-mesh distance measures agreement between two reconstructions.",
            "The low-mesh reference is the fixed triangulated input. Samples estimate percentiles and miss some localized errors.",
            "Edge incidence and Euler characteristic do not certify self-intersection-free or vertex-manifold geometry.",
            "Wall times are one local run per method and exclude evaluation, UV unwrapping, and baking.",
        ],
        "high_snowman": high_rows,
        "high_disagreement_voxel_reference": high_disagreement,
        "low_snowman_voxel_high": snowman_low,
        "low_two_analytic_tori": torus_low,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(compared, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
