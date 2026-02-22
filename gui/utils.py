import numpy as np
import open3d as o3d
import os
import torch
import trimesh
from sklearn.decomposition import PCA


def _points_to_dense_voxels(points: np.ndarray) -> torch.Tensor:
    if points is None or len(points) == 0:
        return torch.zeros((1, 1, 64, 64, 64), dtype=torch.float32, device='cpu')

    points = np.asarray(points, dtype=np.float32)
    points = np.clip(points, -0.5 + 1e-6, 0.5 - 1e-6)
    unique_points = np.floor((points + 0.5) / (1 / 64.0)).astype(np.int32)
    unique_points = np.clip(unique_points, 0, 63)
    unique_points = np.unique(unique_points, axis=0)

    coords_dense = torch.zeros((1, 1, 64, 64, 64), device='cpu', dtype=torch.float32)
    coords_dense[0, 0, unique_points[:, 0], unique_points[:, 1], unique_points[:, 2]] = 1.0
    return coords_dense


def merge_meshes(mesh_list):
    merged = o3d.geometry.TriangleMesh()
    v_offset = 0

    for mesh in mesh_list:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles) + v_offset

        merged.vertices.extend(o3d.utility.Vector3dVector(vertices))
        merged.triangles.extend(o3d.utility.Vector3iVector(triangles))

        v_offset += len(vertices)

    return merged


def voxelgrid_to_open3d(voxels: np.ndarray, threshold=0.5):
    if len(voxels.shape) > 3:
        C, D, H, W = voxels.shape
        flat_feats = voxels.reshape(C, -1).transpose(1,0)
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(flat_feats)
        # Compute feature norm and PCA color std

        # Normalize for RGB
        reduced -= reduced.min(0)
        reduced /= reduced.max(0) + 1e-6

        # Compute norms and color std
        norms = np.linalg.norm(flat_feats, axis=1)
        color_std = np.std(reduced, axis=1)

        # Filter: active voxels with non-trivial color
        mask = (norms > threshold) & (color_std > 1e-3)

        # zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        xx, yy, zz = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        valid_coords = coords[mask]
        valid_colors = reduced[mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_coords.astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(valid_colors.astype(np.float32))
    else:
        coords = np.argwhere(voxels > threshold)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
    return pcd


def save_voxelgrid_as_ply(voxels: np.ndarray, filename: str, threshold=0.5):
    pcd = voxelgrid_to_open3d(voxels, threshold)
    o3d.io.write_point_cloud(filename, pcd)


def voxelize_sq_francis(file_name):
    loaded = trimesh.load(file_name)

    # Point cloud input (e.g., sparse canonical points.ply)
    if isinstance(loaded, trimesh.points.PointCloud):
        return _points_to_dense_voxels(np.asarray(loaded.vertices))

    # Scene input: gather meshes and/or point clouds.
    if isinstance(loaded, trimesh.Scene):
        meshes = []
        points = []
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh) and geom.vertices is not None and len(geom.vertices) > 0 and len(geom.faces) > 0:
                meshes.append(geom)
            elif isinstance(geom, trimesh.points.PointCloud) and geom.vertices is not None and len(geom.vertices) > 0:
                points.append(np.asarray(geom.vertices))

        if meshes:
            mesh = trimesh.util.concatenate(meshes)
            mesh.vertices = np.clip(mesh.vertices, -0.5 + 1e-6, 0.5 - 1e-6)
            voxel_grid = mesh.voxelized(pitch=1 / 64.0)
            return _points_to_dense_voxels(voxel_grid.points)

        if points:
            return _points_to_dense_voxels(np.concatenate(points, axis=0))

        raise ValueError(f"No valid geometry found in scene: {file_name}")

    # Mesh input
    if isinstance(loaded, trimesh.Trimesh):
        if loaded.vertices is None or len(loaded.vertices) == 0:
            raise ValueError(f"Failed to load mesh from {file_name}")
        loaded.vertices = np.clip(loaded.vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        voxel_grid = loaded.voxelized(pitch=1 / 64.0)
        return _points_to_dense_voxels(voxel_grid.points)

    raise ValueError(f"Unsupported geometry type for voxelization: {type(loaded)}")