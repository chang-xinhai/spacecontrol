import argparse
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def add_superquadric_compact_rot_mat(
    scalings: np.ndarray,
    exponents: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    resolution: int = 100,
):
    def f(o, m):
        return np.sign(np.sin(o)) * np.abs(np.sin(o)) ** m

    def g(o, m):
        return np.sign(np.cos(o)) * np.abs(np.cos(o)) ** m

    u = np.linspace(-np.pi, np.pi, resolution, endpoint=True)
    v = np.linspace(-np.pi / 2.0, np.pi / 2.0, resolution, endpoint=True)
    u = np.tile(u, resolution)
    v = np.repeat(v, resolution)
    if np.linalg.det(rotation) < 0:
        u = u[::-1]

    x = scalings[0] * g(v, exponents[0]) * g(u, exponents[1])
    y = scalings[1] * g(v, exponents[0]) * f(u, exponents[1])
    z = scalings[2] * f(v, exponents[0])

    x[:resolution] = 0.0
    x[-resolution:] = 0.0

    vertices = np.concatenate(
        [
            np.expand_dims(x, 1),
            np.expand_dims(y, 1),
            np.expand_dims(z, 1),
        ],
        axis=1,
    )
    vertices = (rotation @ vertices.T).T + translation

    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            triangles.append([i * resolution + j, i * resolution + j + 1, (i + 1) * resolution + j])
            triangles.append([(i + 1) * resolution + j, i * resolution + j + 1, (i + 1) * resolution + (j + 1)])

    for i in range(resolution - 1):
        triangles.append([i * resolution + (resolution - 1), i * resolution, (i + 1) * resolution + (resolution - 1)])
        triangles.append([(i + 1) * resolution + (resolution - 1), i * resolution, (i + 1) * resolution])

    triangles.append([(resolution - 1) * resolution + (resolution - 1), (resolution - 1) * resolution, (resolution - 1)])
    triangles.append([(resolution - 1), (resolution - 1) * resolution, 0])

    return vertices, np.asarray(triangles, dtype=np.int32)


def merge_meshes(vertices_faces_list):
    all_vertices = []
    all_faces = []
    v_offset = 0

    for vertices, faces in vertices_faces_list:
        all_vertices.append(vertices)
        all_faces.append(faces + v_offset)
        v_offset += vertices.shape[0]

    merged_vertices = np.concatenate(all_vertices, axis=0)
    merged_faces = np.concatenate(all_faces, axis=0)
    return merged_vertices, merged_faces


def load_superquadrics_from_npz(file_path: str):
    par_dict = np.load(file_path)
    scales = par_dict["scales"]
    rotations = par_dict["rotations"]
    shapes = par_dict["shapes"]
    translations = par_dict["translations"]

    superquadrics = []
    for idx in range(scales.shape[0]):
        superquadrics.append(
            {
                "scale": scales[idx, :],
                "shape": shapes[idx],
                "rotation": rotations[idx, :],
                "translation": translations[idx, :],
            }
        )
    return superquadrics


def build_spatial_control_mesh(superquadrics, resolution: int = 100):
    meshes = []
    for sq in superquadrics:
        vertices, triangles = add_superquadric_compact_rot_mat(
            sq["scale"], sq["shape"], sq["translation"], sq["rotation"], resolution=resolution
        )
        meshes.append((vertices, triangles))

    verts, faces = merge_meshes(meshes)
    aabb = np.stack([verts.min(0), verts.max(0)])
    center = (aabb[0] + aabb[1]) / 2
    scale = 1 / ((aabb[1] - aabb[0]).max())

    verts = (verts - center) * scale
    return verts, faces, center, scale


def stage_file(out_dir: Path, name: str):
    (out_dir / f"stage_{name}.ok").write_text(f"{time.time()}\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Headless local SpaceControl generation and save.")
    parser.add_argument("--template", type=str, default="gui/superquadrics/chair_sq.npz")
    parser.add_argument("--prompt", type=str, default="chair")
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="generated_assets/local_pipeline")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--t0", type=float, default=6.0)
    parser.add_argument("--sq-resolution", type=int, default=100)
    parser.add_argument("--attn-backend", type=str, default="xformers", choices=["xformers", "flash_attn", "sdpa", "naive"])
    parser.add_argument("--sparse-attn-backend", type=str, default="xformers", choices=["xformers", "flash_attn"])
    parser.add_argument("--spconv-algo", type=str, default="native", choices=["native", "auto", "implicit_gemm"])
    parser.add_argument("--formats", nargs="+", default=["mesh"], choices=["mesh", "gaussian", "radiance_field"])
    parser.add_argument("--save-glb", action="store_true")
    args = parser.parse_args()

    os.environ["ATTN_BACKEND"] = args.attn_backend
    os.environ["SPARSE_ATTN_BACKEND"] = args.sparse_attn_backend
    os.environ["SPCONV_ALGO"] = args.spconv_algo

    from trellis.pipelines import TrellisTextTo3DPipeline
    from trellis.utils import postprocessing_utils

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"{ts}_{args.prompt.replace(' ', '_')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print("[stage] load superquadrics")
    superquadrics = load_superquadrics_from_npz(args.template)
    stage_file(run_dir, "superquadrics_loaded")

    print("[stage] build spatial control mesh")
    merged_vertices, merged_faces, center, scale = build_spatial_control_mesh(superquadrics, resolution=args.sq_resolution)
    spatial_control_mesh_path = run_dir / "spatial_control_mesh.ply"
    trimesh.Trimesh(vertices=merged_vertices, faces=merged_faces, process=False).export(str(spatial_control_mesh_path))

    (run_dir / "spatial_norm.json").write_text(
        json.dumps({"center": center.tolist(), "scale": float(scale)}, indent=2), encoding="utf-8"
    )
    stage_file(run_dir, "spatial_control_ready")

    print("[stage] load pipeline")
    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda()
    stage_file(run_dir, "pipeline_loaded")

    image_prompt = None
    if args.image:
        with open(args.image, "rb") as f:
            image_prompt = Image.open(BytesIO(f.read()))
    print("[stage] run generation")
    outputs = pipeline.run(
        args.prompt,
        image_prompt,
        seed=args.seed,
        sparse_structure_sampler_params={
            "steps": args.steps,
            "cfg_strength": args.cfg,
            "t0_idx_value": args.t0,
            "spatial_control_mesh_path": str(spatial_control_mesh_path),
        },
        formats=args.formats,
    )
    stage_file(run_dir, "generation_done")

    if "mesh" in outputs and len(outputs["mesh"]) > 0:
        mesh_out = outputs["mesh"][0]
        vertices = mesh_out.vertices.detach().float().cpu().numpy()
        faces = mesh_out.faces.detach().long().cpu().numpy()
        vertices = vertices / scale + center
        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh_trimesh.export(str(run_dir / "generated_mesh.ply"))
        mesh_trimesh.export(str(run_dir / "generated_mesh.obj"))
        stage_file(run_dir, "mesh_saved")

    if "gaussian" in outputs and len(outputs["gaussian"]) > 0:
        outputs["gaussian"][0].save_ply(str(run_dir / "generated_gaussian.ply"))
        stage_file(run_dir, "gaussian_saved")

    if args.save_glb:
        if not ("mesh" in outputs and "gaussian" in outputs and len(outputs["mesh"]) > 0 and len(outputs["gaussian"]) > 0):
            raise ValueError("--save-glb requires both mesh and gaussian outputs. Use --formats mesh gaussian")
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=0.95,
            texture_size=1024,
        )
        glb.apply_scale(1 / scale)
        glb.apply_translation(center)
        glb.export(str(run_dir / "generated.glb"))
        stage_file(run_dir, "glb_saved")

    print(f"[done] outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
