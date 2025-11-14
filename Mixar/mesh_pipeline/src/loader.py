# src/loader.py
"""
Dataset discovery and safe loading helpers.

Functions:
- find_mesh_files(root, exts=(".obj",".ply",".stl"), ignore_dotfiles=True)
- safe_load_mesh(path, prefer_trimesh=True) -> (mesh_obj, backend)
- load_mesh_info(path, prefer_trimesh=True) -> dict with vertices, faces, stats
- iterate_meshes(root, ...) -> generator yielding (path, info_dict)
"""

from pathlib import Path
from typing import List, Tuple, Optional, Generator, Dict, Any
import numpy as np

# Try to import both libraries; use what is available
try:
    import trimesh
except Exception:
    trimesh = None

try:
    import open3d as o3d
except Exception:
    o3d = None


def find_mesh_files(root: str or Path,
                    exts: Tuple[str, ...] = (".obj", ".ply", ".stl"),
                    ignore_dotfiles: bool = True,
                    recursive: bool = True) -> List[Path]:
    """
    Find mesh files under `root` with given extensions.

    Args:
        root: directory to scan
        exts: tuple of file extensions (lowercase)
        ignore_dotfiles: skip files starting with '.' or '._'
        recursive: whether to scan subdirectories

    Returns:
        sorted list of pathlib.Path objects
    """
    root_p = Path(root)
    if not root_p.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")
    pattern = "**/*" if recursive else "*"
    files = []
    for p in root_p.glob(pattern):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        name = p.name
        if ignore_dotfiles and (name.startswith("._") or name.startswith(".")):
            continue
        files.append(p)
    # sort for deterministic order
    files = sorted(files, key=lambda x: str(x))
    return files


def safe_load_mesh(path: str or Path, prefer_trimesh: bool = True):
    """
    Load a mesh using trimesh (preferred) or open3d (fallback). Returns a tuple (mesh, backend)
    where `mesh` is:
      - trimesh.Trimesh instance if backend == 'trimesh'
      - open3d.geometry.TriangleMesh instance if backend == 'open3d'

    Note: this function does NOT process/repair meshes; it loads raw data.
    """
    p = Path(path)
    if prefer_trimesh and trimesh is not None:
        try:
            mesh = trimesh.load(str(p), process=False)
            # If a Scene, concatenate geometries
            if isinstance(mesh, trimesh.Scene):
                geoms = [g for g in mesh.geometry.values()]
                if len(geoms) == 0:
                    raise ValueError("Trimesh Scene contains no geometries")
                mesh = trimesh.util.concatenate(geoms)
            return mesh, "trimesh"
        except Exception:
            # fall back to open3d if available
            pass

    if o3d is not None:
        try:
            mesh_o3d = o3d.io.read_triangle_mesh(str(p))
            if mesh_o3d.is_empty():
                raise ValueError("Open3D loaded an empty mesh")
            return mesh_o3d, "open3d"
        except Exception:
            pass

    # Last attempt: try trimesh even if prefer_trimesh False (if trimesh exists)
    if trimesh is not None:
        mesh = trimesh.load(str(p), process=False)
        if isinstance(mesh, trimesh.Scene):
            geoms = [g for g in mesh.geometry.values()]
            mesh = trimesh.util.concatenate(geoms)
        return mesh, "trimesh"

    raise RuntimeError("No mesh loader available (trimesh/open3d not found or loading failed).")


def mesh_to_numpy(mesh_obj, backend: str):
    """
    Convert a loaded mesh to (vertices, faces) as numpy arrays.

    For trimesh: mesh_obj.vertices, mesh_obj.faces
    For open3d : np.asarray(mesh_obj.vertices), np.asarray(mesh_obj.triangles)
    """
    if backend == "trimesh":
        verts = np.asarray(mesh_obj.vertices, dtype=float)
        faces = np.asarray(mesh_obj.faces, dtype=np.int64) if hasattr(mesh_obj, "faces") else None
    elif backend == "open3d":
        verts = np.asarray(mesh_obj.vertices, dtype=float)
        faces = np.asarray(mesh_obj.triangles, dtype=np.int64) if hasattr(mesh_obj, "triangles") else None
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return verts, faces


def compute_vertex_stats(vertices: np.ndarray) -> Dict[str, Any]:
    v = np.asarray(vertices, dtype=float)
    return {
        "n_vertices": int(v.shape[0]),
        "min": v.min(axis=0).tolist(),
        "max": v.max(axis=0).tolist(),
        "mean": v.mean(axis=0).tolist(),
        "std": v.std(axis=0).tolist()
    }


def load_mesh_info(path: str or Path, prefer_trimesh: bool = True) -> Dict[str, Any]:
    """
    Load mesh and return a dict containing:
      - path: str
      - backend: 'trimesh' or 'open3d'
      - vertices: (N,3) numpy array
      - faces: (M,3) numpy array or None
      - stats: dict with n_vertices, min, max, mean, std
    """
    mesh_obj, backend = safe_load_mesh(path, prefer_trimesh=prefer_trimesh)
    verts, faces = mesh_to_numpy(mesh_obj, backend)
    stats = compute_vertex_stats(verts)
    return {
        "path": str(path),
        "backend": backend,
        "vertices": verts,
        "faces": faces,
        "stats": stats
    }


def iterate_meshes(root: str or Path,
                   exts: Tuple[str, ...] = (".obj", ".ply", ".stl"),
                   ignore_dotfiles: bool = True,
                   prefer_trimesh: bool = True,
                   recursive: bool = True) -> Generator[Dict[str, Any], None, None]:
    """
    Generator that yields mesh info dicts for every mesh found under root.
    Useful for streaming large datasets without loading all at once.
    """
    files = find_mesh_files(root, exts=exts, ignore_dotfiles=ignore_dotfiles, recursive=recursive)
    for p in files:
        try:
            info = load_mesh_info(p, prefer_trimesh=prefer_trimesh)
            yield info
        except Exception as e:
            # yield an error record instead of raising to allow batch processing to continue
            yield {"path": str(p), "error": str(e)}
