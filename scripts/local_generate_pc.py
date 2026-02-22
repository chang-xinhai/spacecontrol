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


def load_point_cloud_vertices(file_path: str) -> np.ndarray:
    loaded = trimesh.load(file_path)

    if isinstance(loaded, trimesh.points.PointCloud):
        points = np.asarray(loaded.vertices, dtype=np.float64)
    elif isinstance(loaded, trimesh.Scene):
        all_points = []
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.points.PointCloud) and geom.vertices is not None and len(geom.vertices) > 0:
                all_points.append(np.asarray(geom.vertices, dtype=np.float64))
            elif isinstance(geom, trimesh.Trimesh) and geom.vertices is not None and len(geom.vertices) > 0:
                all_points.append(np.asarray(geom.vertices, dtype=np.float64))
        if not all_points:
            raise ValueError(f"No valid point-like geometry found in: {file_path}")
        points = np.concatenate(all_points, axis=0)
    elif isinstance(loaded, trimesh.Trimesh):
        points = np.asarray(loaded.vertices, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported geometry type for point cloud template: {type(loaded)}")

    if points.size == 0:
        raise ValueError(f"Template point cloud is empty: {file_path}")
    return points


def canonicalize_points(points: np.ndarray):
    aabb = np.stack([points.min(0), points.max(0)])
    center = (aabb[0] + aabb[1]) / 2
    extent = (aabb[1] - aabb[0]).max()
    if extent <= 0:
        raise ValueError("Template point cloud has degenerate bounds; cannot normalize")
    scale = 1.0 / float(extent)
    canonical_points = (points - center) * scale
    return canonical_points, center, scale


def stage_file(out_dir: Path, name: str):
    (out_dir / f"stage_{name}.ok").write_text(f"{time.time()}\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Headless local SpaceControl generation from a point-cloud template and save.")
    parser.add_argument("--template", type=str, default="data/GSO_example/BEDROOM_NEO/vggt/view_4/canonical/sparse/points.ply")
    parser.add_argument("--canonicalize-template", action="store_true")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="generated_assets/local_pipeline")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--t0", type=float, default=6.0)
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

    print("[stage] load template point cloud")
    template_path = Path(args.template)
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    center = np.zeros(3, dtype=np.float64)
    scale = 1.0

    if args.canonicalize_template:
        points = load_point_cloud_vertices(str(template_path))
        canonical_points, center, scale = canonicalize_points(points)
        spatial_control_pc_path = run_dir / "spatial_control_points.ply"
        trimesh.points.PointCloud(vertices=canonical_points).export(str(spatial_control_pc_path))
    else:
        spatial_control_pc_path = template_path

    stage_file(run_dir, "template_pointcloud_ready")

    (run_dir / "spatial_norm.json").write_text(
        json.dumps({"center": center.tolist(), "scale": float(scale)}, indent=2), encoding="utf-8"
    )

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
            "spatial_control_mesh_path": str(spatial_control_pc_path),
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
