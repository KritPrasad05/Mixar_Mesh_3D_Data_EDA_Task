# src/pipeline.py
"""
Main processing pipeline for mesh normalization, quantization, reconstruction, and evaluation.

Functions:
- process_mesh(path, out_root, normalizer='minmax', n_bins=1024)
- process_all_meshes(data_root, out_root, normalizer='minmax', n_bins=1024)
"""

from pathlib import Path
import numpy as np
import json
import trimesh

from src.loader import load_mesh_info, find_mesh_files
from src.transforms import MinMaxNormalizer, UnitSphereNormalizer, Quantizer
from src.io import save_mesh, save_metadata
from src.metrics import compute_errors, save_error_metrics, plot_error_per_axis
from src.viz import render_mesh, compare_meshes_plot


def process_mesh(path: str or Path,
                 out_root: str or Path,
                 normalizer: str = 'minmax',
                 n_bins: int = 1024,
                 prefer_trimesh: bool = True) -> dict:
    """
    Process a single mesh file end-to-end:
        load → normalize → quantize → reconstruct → compute metrics → visualize → save outputs
    """
    p = Path(path)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n[PROCESS] {p.name} | normalizer={normalizer} | bins={n_bins}")

    # 1. Load mesh
    info = load_mesh_info(p, prefer_trimesh=prefer_trimesh)
    verts = info["vertices"]
    faces = info["faces"]

    # 2. Normalize
    if normalizer == "minmax":
        norm = MinMaxNormalizer().fit(verts)
        v_norm = norm.transform(verts)
    elif normalizer == "unitsphere":
        norm = UnitSphereNormalizer().fit(verts)
        v_norm = norm.transform(verts)
        # map [-1,1] to [0,1] for quantization
        v_norm = (v_norm + 1.0) / 2.0
    else:
        raise ValueError("normalizer must be 'minmax' or 'unitsphere'")

    # 3. Quantize + Dequantize
    quantizer = Quantizer(n_bins=n_bins)
    q = quantizer.transform(v_norm)
    v_deq = quantizer.inverse_transform(q)

    # For unitsphere, map back [0,1] → [-1,1]
    if normalizer == "unitsphere":
        v_deq = v_deq * 2.0 - 1.0

    # 4. Denormalize (reconstruction)
    v_recon = norm.inverse_transform(v_deq)

    # 5. Compute metrics
    errors = compute_errors(verts, v_recon)

    # 6. Save reconstructed mesh and metadata
    recon_dir = out_root / "reconstructed"
    recon_dir.mkdir(parents=True, exist_ok=True)
    recon_path = recon_dir / f"{p.stem}_recon.obj"

    # Load original topology to preserve faces
    orig_mesh = trimesh.load(str(p), process=False)
    if isinstance(orig_mesh, trimesh.Scene):
        orig_mesh = trimesh.util.concatenate(list(orig_mesh.geometry.values()))
    save_mesh(orig_mesh, str(recon_path), vertices=v_recon, file_type="obj")

    # Save metadata and error metrics
    meta = {
        "source": str(p),
        "normalizer": norm.get_metadata(),
        "quantizer": quantizer.get_metadata(),
        "errors": errors
    }
    meta_path = out_root / "metadata" / f"{p.stem}_meta.json"
    save_metadata(meta, str(meta_path))

    metrics_path = out_root / "metrics" / f"{p.stem}_metrics.json"
    save_error_metrics(errors, str(metrics_path))

    # 7. Plot error per axis
    plot_path = out_root / "metrics" / f"{p.stem}_error.png"
    plot_error_per_axis(errors, str(plot_path), title=f"{p.stem} Reconstruction Error")

    # 8. Visualize original vs reconstructed
    render_dir = out_root / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    try:
        render_mesh(orig_mesh, str(render_dir / f"{p.stem}_orig.png"))
        recon_mesh = trimesh.load(str(recon_path), process=False)
        if isinstance(recon_mesh, trimesh.Scene):
            recon_mesh = trimesh.util.concatenate(list(recon_mesh.geometry.values()))
        render_mesh(recon_mesh, str(render_dir / f"{p.stem}_recon.png"))
        compare_meshes_plot(orig_mesh, recon_mesh, str(render_dir / f"{p.stem}_compare.png"))
    except Exception as e:
        print("Render warning:", e)

    # 9. Save quantized npz (for inspection)
    q_dir = out_root / "quantized"
    q_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(q_dir / f"{p.stem}_q.npz", quant=q)

    print(f"[DONE] {p.name} | MSE={errors['mse']:.6e} | MAE={errors['mae']:.6e}")
    return {
        "mesh": str(p),
        "reconstructed": str(recon_path),
        "metrics": errors,
        "meta_path": str(meta_path),
        "error_plot": str(plot_path)
    }


def process_all_meshes(data_root: str or Path,
                       out_root: str or Path,
                       normalizer: str = 'minmax',
                       n_bins: int = 1024) -> dict:
    """
    Process all meshes in a folder.
    Returns a dict mapping filenames to results.
    """
    data_root = Path(data_root)
    out_root = Path(out_root)
    files = find_mesh_files(data_root)
    print(f"\nFound {len(files)} meshes in {data_root}")

    summary = {}
    for f in files:
        try:
            res = process_mesh(f, out_root, normalizer=normalizer, n_bins=n_bins)
            summary[f.name] = res
        except Exception as e:
            print(f"[ERROR] {f.name} failed:", e)
            summary[f.name] = {"error": str(e)}

    # Save a global summary JSON
    summary_path = out_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\nAll meshes processed.")
    print("Summary saved to:", summary_path)
    return summary
# --- ADAPTIVE BONUS SUPPORT ----------------------------------------------------
import json
import numpy as np
from pathlib import Path
from src.transforms import MinMaxNormalizer, Quantizer
from src.metrics import compute_errors
from src.loader import load_mesh_info
from src.metrics import save_error_metrics
from src.viz import improved_render
from scipy.spatial import cKDTree
import trimesh

def vertex_density(vertices, k=16):
    """Local vertex density (inverse mean neighbor distance)."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape[0] <= k:
        md = np.linalg.norm(vertices - vertices.mean(axis=0), axis=1)
        return 1.0 / (md + 1e-12), md
    tree = cKDTree(vertices)
    dists, _ = tree.query(vertices, k=k+1)
    md = dists[:, 1:].mean(axis=1)
    return 1.0 / (md + 1e-12), md


def adaptive_quantize_roundtrip(vertices, faces, bins_low=256, bins_med=1024,
                                bins_high=4096, k=16):
    """Simplified adaptive quantization prototype."""
    mn = MinMaxNormalizer().fit(vertices)
    v_norm = mn.transform(vertices)
    dens, _ = vertex_density(vertices, k=k)
    dmin, dmax = float(dens.min()), float(dens.max())
    dens_norm = (dens - dmin) / (dmax - dmin + 1e-12)
    q1, q2 = np.quantile(dens_norm, [0.33, 0.66])
    bins_per_vertex = np.full(len(vertices), bins_med, int)
    bins_per_vertex[dens_norm <= q1] = bins_low
    bins_per_vertex[dens_norm > q2] = bins_high

    q_all = np.zeros_like(v_norm, int)
    deq_all = np.zeros_like(v_norm, float)
    for b in np.unique(bins_per_vertex):
        idx = np.where(bins_per_vertex == b)[0]
        if idx.size == 0: continue
        v_sub = v_norm[idx]
        q_sub = np.floor(v_sub * (b - 1)).astype(int)
        q_sub = np.clip(q_sub, 0, b - 1)
        deq_sub = q_sub.astype(float) / (b - 1)
        q_all[idx] = q_sub
        deq_all[idx] = deq_sub

    v_recon = mn.inverse_transform(deq_all)
    return v_recon, bins_per_vertex


def process_mesh_adaptive(path, out_root, bins_low=256, bins_med=1024,
                          bins_high=4096, k=16, n_rot=4):
    """
    Adaptive quantization prototype (bonus). Evaluates multiple rotations and
    computes mean MSE for adaptive vs uniform.
    """
    path = Path(path)
    out_root = Path(out_root) / "adaptive_bonus"
    out_root.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(str(path), process=False)
    faces = getattr(mesh, "faces", None)

    uniform_mse, adaptive_mse = [], []
    for i in range(n_rot):
        R = trimesh.transformations.random_rotation_matrix()
        m2 = mesh.copy()
        m2.apply_transform(R)
        verts = m2.vertices

        # uniform baseline
        mn = MinMaxNormalizer().fit(verts)
        v_norm = mn.transform(verts)
        q = Quantizer(n_bins=1024).transform(v_norm)
        v_deq = Quantizer(n_bins=1024).inverse_transform(q)
        v_recon = mn.inverse_transform(v_deq)
        err_u = compute_errors(verts, v_recon)
        uniform_mse.append(err_u["mse"])

        # adaptive
        v_recon2, bins = adaptive_quantize_roundtrip(verts, faces, bins_low, bins_med, bins_high, k)
        err_a = compute_errors(verts, v_recon2)
        adaptive_mse.append(err_a["mse"])

    summary = {
        "mesh": path.name,
        "uniform_mse_mean": float(np.mean(uniform_mse)),
        "adaptive_mse_mean": float(np.mean(adaptive_mse)),
        "uniform_mse_list": [float(x) for x in uniform_mse],
        "adaptive_mse_list": [float(x) for x in adaptive_mse],
        "bins_low": bins_low, "bins_med": bins_med, "bins_high": bins_high
    }
    (out_root / f"{path.stem}_adaptive_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ADAPTIVE DONE] {path.name} | uniform={summary['uniform_mse_mean']:.2e} "
          f"| adaptive={summary['adaptive_mse_mean']:.2e}")
    return summary