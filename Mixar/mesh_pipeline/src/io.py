"""
src/io.py

Mesh I/O helpers using trimesh.

Functions:
- load_mesh(path, process=False): load a mesh and return trimesh.Trimesh
- save_mesh(mesh, out_path, vertices=None, file_type=None): save mesh (optionally with replaced vertices)
- save_metadata(meta, out_json_path): save metadata (normalization params) to JSON
- load_metadata(in_json_path): load metadata JSON
- render_and_save_image(mesh, out_image_path, resolution=(1024,768)): render mesh to PNG bytes and write file
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import trimesh


def load_mesh(path: str, process: bool = False) -> trimesh.Trimesh:
    """
    Load a mesh from disk using trimesh.

    Args:
        path: file path to .obj/.ply/.stl etc.
        process: if True, allow trimesh to process/repair the mesh. For exact vertex preservation use False.

    Returns:
        trimesh.Trimesh instance
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    mesh = trimesh.load(str(p), process=process)
    # If file contains a Scene with multiple geometries, try to combine them
    if isinstance(mesh, trimesh.Scene):
        # convert to single mesh by concatenating
        geom_list = [g for g in mesh.geometry.values()]
        if len(geom_list) == 0:
            raise ValueError("Loaded scene contains no geometries.")
        # Merge into a single Trimesh (this may reindex faces)
        mesh = trimesh.util.concatenate(geom_list)
    return mesh


def save_mesh(mesh: trimesh.Trimesh, out_path: str,
              vertices: Optional[np.ndarray] = None,
              file_type: Optional[str] = None) -> None:
    """
    Save a mesh to disk. Optionally replace vertices before saving.

    Args:
        mesh: source trimesh.Trimesh (will be copied internally)
        out_path: output filename (extension determines format if file_type not provided)
        vertices: optional (N,3) array of vertices to set before saving
        file_type: optional explicit format (e.g., 'ply' or 'obj'); if None inferred from out_path
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    mesh_to_save = mesh.copy()
    if vertices is not None:
        vertices = np.asarray(vertices)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must be (N,3) array.")
        # Ensure same number of vertices if topology must remain identical:
        if vertices.shape[0] != mesh_to_save.vertices.shape[0]:
            # Allow saving anyway but warn user
            print(f"Warning: replacing vertices length {vertices.shape[0]} != original {mesh_to_save.vertices.shape[0]}")
        mesh_to_save.vertices = vertices

    # Choose format
    ext = file_type if file_type is not None else p.suffix.lstrip('.').lower()
    if ext == '':
        raise ValueError("Output path must have a file extension or you must set file_type.")

    mesh_to_save.export(file_obj=str(p), file_type=ext)


def save_metadata(meta: Dict[str, Any], out_json_path: str) -> None:
    """
    Save metadata (e.g. normalization params) as JSON.
    Handles nested numpy types recursively.

    Args:
        meta: dictionary containing metadata
        out_json_path: path to write JSON
    """
    import json
    import numpy as np
    from pathlib import Path

    def make_serializable(obj):
        """Recursively convert NumPy types inside nested structures."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    p = Path(out_json_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable = make_serializable(meta)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)


def load_metadata(in_json_path: str) -> Dict[str, Any]:
    """
    Load metadata JSON and return as dict.

    Args:
        in_json_path: path to metadata json
    """
    p = Path(in_json_path)
    if not p.exists():
        raise FileNotFoundError(in_json_path)
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    # Convert lists back to numpy arrays where appropriate (heuristic)
    for k, v in list(d.items()):
        if isinstance(v, list) and len(v) > 0 and all(isinstance(x, (int, float)) for x in v):
            d[k] = np.asarray(v, dtype=float)
    return d


def render_and_save_image(mesh: trimesh.Trimesh, out_image_path: str, resolution: Tuple[int, int] = (1024, 768)) -> None:
    """
    Render mesh with trimesh.scene and save PNG. Returns early if rendering backend isn't available.

    Args:
        mesh: trimesh.Trimesh
        out_image_path: path to write PNG
        resolution: (width, height)
    """
    p = Path(out_image_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    scene = mesh.scene()
    # trimesh Scene.save_image may return bytes or raise if no renderer present
    try:
        img_bytes = scene.save_image(resolution=resolution, visible=True)
    except Exception as e:
        print("Warning: trimesh scene save_image failed:", str(e))
        print("You may need a rendering backend (pyglet) or use an alternative plotting method.")
        return

    if img_bytes is None:
        print("Warning: scene.save_image returned None (no backend).")
        return

    with open(p, "wb") as f:
        f.write(img_bytes)
