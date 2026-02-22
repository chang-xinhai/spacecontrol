import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def _load_mesh(mesh_path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError(f"No geometry found in scene: {mesh_path}")
        meshes = []
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0 and len(geom.faces) > 0:
                meshes.append(geom)
        if not meshes:
            raise ValueError(f"No triangle meshes found in scene: {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Unsupported mesh type: {type(loaded)}")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")
    return mesh


def _canonical_normalize(vertices: np.ndarray):
    aabb = np.stack([vertices.min(0), vertices.max(0)])
    center = (aabb[0] + aabb[1]) / 2.0
    extent = (aabb[1] - aabb[0]).max()
    if extent <= 0:
        raise ValueError("Degenerate mesh extent; cannot normalize.")
    scale = 1.0 / float(extent)
    vertices_canonical = (vertices - center) * scale
    return vertices_canonical, center, scale


def _rotation_y_up_to_axis(axis: str) -> np.ndarray:
    axis = axis.upper()
    if axis == "Y":
        return np.eye(3, dtype=np.float64)
    if axis == "Z":
        # Rotate +90° around X so +Y -> +Z.
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    if axis == "X":
        # Rotate -90° around Z so +Y -> +X.
        return np.array(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported axis '{axis}'. Expected one of: X, Y, Z")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GLB/mesh to canonical reference PLY expected by SpaceControl spatial control voxelization."
    )
    parser.add_argument("--input", type=str, default="data/GSO_example/BEDROOM_NEO/model.glb")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--meta", type=str, default="")
    parser.add_argument("--axis", type=str, default="Z", choices=["X", "Y", "Z", "x", "y", "z"])
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input mesh not found: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_canonical.ply")

    if args.meta:
        meta_path = Path(args.meta)
    else:
        meta_path = output_path.with_name(f"{output_path.stem}_norm.json")

    mesh = _load_mesh(str(input_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    vertices_canonical, center, scale = _canonical_normalize(vertices)
    rot = _rotation_y_up_to_axis(args.axis)
    vertices_canonical = vertices_canonical @ rot.T
    canonical_mesh = trimesh.Trimesh(vertices=vertices_canonical, faces=faces, process=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_mesh.export(str(output_path))

    meta = {
        "input": str(input_path),
        "output": str(output_path),
        "axis": args.axis.upper(),
        "center": center.tolist(),
        "scale": float(scale),
        "note": "Applied canonical alignment used by local_generate.py, then rotated from assumed Y-up input to requested output up-axis",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[done] canonical mesh saved to: {output_path}")
    print(f"[done] normalization metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
