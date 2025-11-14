# src/viz.py
"""
Viz helpers with improved offscreen rendering.

Functions:
- improved_render(mesh, out_path, resolution=(1600,1200), bgcolor=(1,1,1,1), smooth=True)
- render_mesh(mesh, out_path, resolution=(1024,768))  # kept for compatibility
- compare_meshes_plot(mesh1, mesh2, out_path, resolution=(1600,900))
"""
from pathlib import Path
import numpy as np

# try imports
try:
    import pyrender
except Exception:
    pyrender = None

try:
    import trimesh
except Exception:
    trimesh = None

# matplotlib fallback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _matplotlib_point_cloud(v, out_path, title=None, dpi=200):
    """Simple scatter fallback (good for sparse / debug)"""
    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=1)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), dpi=dpi)
    plt.close(fig)
    return True


def _trimesh_render(mesh, out_path, resolution=(1600, 1200), bgcolor=(1, 1, 1, 1)):
    """Use trimesh.scene.save_image (needs pyglet)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    scene = mesh.scene()
    try:
        img = scene.save_image(resolution=resolution, visible=True, background=bgcolor)
        if img is None:
            return False
        with open(p, "wb") as f:
            f.write(img)
        return True
    except Exception:
        return False


def _pyrender_render(mesh, out_path, resolution=(1600, 1200), bgcolor=(1.0, 1.0, 1.0, 1.0), smooth=True):
    """
    Render using pyrender for nicer lighting and shading.
    Accepts a trimesh.Trimesh or pyrender.MeshNode input.
    """
    if pyrender is None:
        return False
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Ensure trimesh object
    if hasattr(mesh, "scene") and hasattr(mesh, "vertices"):
        tm = mesh
    else:
        # try to convert
        try:
            tm = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces))
        except Exception:
            return False

    # Create pyrender scene
    scene = pyrender.Scene(bg_color=np.array(bgcolor[:3], dtype=float), ambient_light=(0.3, 0.3, 0.3))
    try:
        # create pyrender mesh with smooth shading if requested
        try:
            pm = pyrender.Mesh.from_trimesh(tm, smooth=smooth)
        except Exception:
            pm = pyrender.Mesh.from_trimesh(tm, smooth=False)
        node = scene.add(pm)

        # Add lights
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(light, pose=np.eye(4))
        # add a second light slightly offset
        pose = np.eye(4)
        pose[0, 3] = 1.0
        pose[1, 3] = 1.0
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), pose=pose)

        # camera: try an automatic camera centered on mesh
        bbox = tm.bounds
        center = tm.centroid
        size = np.max(bbox[1] - bbox[0])
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        cam_pose = np.array([
            [1.0, 0.0, 0.0, center[0] + size * 1.2],
            [0.0, 1.0, 0.0, center[1] + size * 0.6],
            [0.0, 0.0, 1.0, center[2] + size * 0.8],
            [0.0, 0.0, 0.0, 1.0],
        ])
        scene.add(camera, pose=cam_pose)

        # Renderer
        r = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
        color, depth = r.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
        r.delete()
        # write color
        import imageio
        imageio.imsave(str(p), color)
        return True
    except Exception:
        return False


def improved_render(mesh, out_path: str, resolution=(1600, 1200), bgcolor=(1, 1, 1, 1), smooth=True):
    """
    High-quality offscreen render that tries pyrender -> trimesh -> matplotlib.
    Returns True if an image was written.
    """
    # prefer pyrender
    if pyrender is not None:
        ok = _pyrender_render(mesh, out_path, resolution=resolution, bgcolor=bgcolor, smooth=smooth)
        if ok:
            return True

    # then trimesh
    ok = _trimesh_render(mesh, out_path, resolution=resolution, bgcolor=bgcolor)
    if ok:
        return True

    # fallback scatter
    try:
        v = mesh.vertices if hasattr(mesh, "vertices") else np.asarray(mesh)
        return _matplotlib_point_cloud(np.asarray(v), out_path, title=None, dpi=200)
    except Exception:
        return False


# keep old names for compatibility
def render_mesh(mesh, out_path: str, resolution=(1024, 768)):
    return improved_render(mesh, out_path, resolution=resolution, smooth=True)


def compare_meshes_plot(mesh1, mesh2, out_path: str, resolution=(1600, 900)):
    """
    Generate side-by-side images using improved_render if pyrender saves images, else use matplotlib scatter.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Try create two temp images and combine
    tmp1 = p.with_suffix(".left.png")
    tmp2 = p.with_suffix(".right.png")
    ok1 = improved_render(mesh1, tmp1, resolution=(resolution[0]//2, resolution[1]), smooth=True)
    ok2 = improved_render(mesh2, tmp2, resolution=(resolution[0]//2, resolution[1]), smooth=True)
    if ok1 and ok2:
        # combine horizontally
        from PIL import Image
        im1 = Image.open(tmp1)
        im2 = Image.open(tmp2)
        new = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), (255,255,255))
        new.paste(im1, (0, 0))
        new.paste(im2, (im1.width, 0))
        new.save(p)
        try:
            tmp1.unlink()
            tmp2.unlink()
        except Exception:
            pass
        return True
    else:
        # fallback matplotlib combined scatter
        v1 = mesh1.vertices if hasattr(mesh1, "vertices") else np.asarray(mesh1)
        v2 = mesh2.vertices if hasattr(mesh2, "vertices") else np.asarray(mesh2)
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(v1[:,0], v1[:,1], v1[:,2], s=1)
        ax1.set_title("Original Mesh")
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(v2[:,0], v2[:,1], v2[:,2], s=1)
        ax2.set_title("Reconstructed Mesh")
        plt.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        return True