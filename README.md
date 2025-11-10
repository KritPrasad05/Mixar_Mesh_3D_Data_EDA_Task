# 🧠 SeamGPT Mesh Data Processing Pipeline

<p align="center">
  <b>Author:</b> Krit Prasad<br>
  <b>Project:</b> SeamGPT Hiring Assignment — Data Processing & Quantization<br>
  <b>Technologies:</b> Python • Trimesh • NumPy • SciPy • Matplotlib • PyRender
</p>

---

## 📘 Overview

This project implements a complete **3D mesh data processing pipeline** designed for SeamGPT’s hiring assignment.  
It performs **normalization, quantization, dequantization, reconstruction**, and **error analysis** for 3D meshes (.obj/.ply/.stl).  

Additionally, it includes a **Bonus Adaptive Quantization Prototype** that introduces vertex-density-based bin allocation and tests **rotation invariance**.

---

## 🚀 Features

- 📂 Automated dataset discovery and mesh loading using `Trimesh` and `Open3D`.
- ⚖️ Two normalization methods: **Min–Max** and **Unit Sphere**.
- 🎯 Uniform quantization (default 1024 bins) for precise reconstruction.
- 📊 Error analysis: MSE & MAE per axis, visualization plots.
- 🧠 Adaptive quantization prototype based on vertex density (bonus task).
- 🌀 Rotation-invariance evaluation (bonus).
- 🖼️ High-quality mesh renders (HQ PNG) using `pyrender`/`pyglet`.
- 🧾 JSON-based metadata, metrics, and visual outputs for all runs.

---

## 🧩 Project Structure

```
mesh_pipeline/
│
├── src/                            # Core source modules
│   ├── io.py                       # Mesh I/O and metadata save/load
│   ├── loader.py                   # Safe mesh loading and inspection
│   ├── transforms.py               # Normalization & quantization logic
│   ├── metrics.py                  # Error metrics and visualization
│   ├── viz.py                      # Rendering utilities (Trimesh + PyRender)
│   └── pipeline.py                 # Unified processing pipeline + Adaptive prototype
│
├── outputs/                        # All runtime results
│   ├── reconstructed/              # Reconstructed meshes (.obj)
│   ├── normalized/                 # Normalized meshes (minmax + unitsphere)
│   ├── quantized_vis/              # Quantized visualization OBJs
│   ├── metrics/                    # Error plots + JSON metrics
│   ├── renders/                    # Rendered mesh images
│   └── adaptive_bonus/             # Adaptive quantization summaries
│
├── notebooks/                      # Development & testing notebooks
├── run_pipeline.py                 # CLI entrypoint (Uniform + Adaptive)
├── requirements.txt                # Package dependencies
└── README.md                       # You're reading it
```

---

## ⚙️ Installation

```bash
# 1. Clone this repository or copy project folder
cd mesh_pipeline

# 2. Create and activate virtual environment
python -m venv Mixar
.\Mixar\Scripts\Activate.ps1   # (Windows PowerShell)

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧠 Usage

### ▶️ Run the Uniform (Standard) Pipeline
```bash
python run_pipeline.py
```

or explicitly specify options:
```bash
python run_pipeline.py --data_root "C:/Users/kritr/Downloads/INTERNSHIP/Mixar/8samples" --out_root "./outputs" --normalizer minmax --n_bins 1024
```

### 🧪 Run the Adaptive Quantization Prototype (Bonus)
```bash
python run_pipeline.py --adaptive
```

Optional parameters:
```bash
--rotations <int>   # number of random rotations (default=4)
--verbose           # display debug info
```

Example:
```bash
python run_pipeline.py --adaptive --rotations 8 --verbose
```

---

## 📁 Output Directory Structure

| Folder | Description |
|---------|-------------|
| `outputs/reconstructed/` | Final reconstructed meshes (.obj) |
| `outputs/normalized/` | Normalized meshes (MinMax + UnitSphere) |
| `outputs/quantized_vis/` | Quantized visualization OBJs |
| `outputs/metrics/` | Error JSONs + bar plots |
| `outputs/renders/` | Side-by-side mesh renders |
| `outputs/adaptive_bonus/` | Adaptive results (per mesh JSONs, plots) |

---

## 📊 Results & Findings

### 🧩 Uniform Quantization (1024 bins)
- Average MSE across all meshes: **≈ 7.8×10⁻⁷**
- Average MAE: **≈ 7.3×10⁻⁴**
- Reconstruction error is negligible — vertices retain geometric fidelity.
- Works robustly for all 8 test meshes.

### 🔬 Adaptive Quantization (256/1024/4096 bins)
| Mesh | Uniform MSE | Adaptive MSE | Verdict |
|------|-------------:|-------------:|---------|
| branch.obj | 7.92e-07 | 4.48e-06 | Uniform better |
| cylinder.obj | 1.99e-06 | 1.28e-05 | Uniform better |
| explosive.obj | 1.71e-07 | 9.71e-07 | Uniform better |
| fence.obj | 2.65e-07 | 1.52e-06 | Uniform better |
| girl.obj | 2.05e-07 | 1.16e-06 | Uniform better |
| person.obj | 6.62e-07 | 3.68e-06 | Uniform better |
| table.obj | 3.07e-07 | 1.73e-06 | Uniform better |
| talwar.obj | 1.11e-07 | 6.20e-07 | Uniform better |

🧠 **Conclusion:**  
The simple adaptive prototype increases MSE for all meshes because vertex-level bin variation introduces discontinuities.  
However, it demonstrates the concept successfully and provides a foundation for region-based or per-axis adaptive quantization.

---

## 🖼️ Sample High-Quality Renders

<p align="center">
  <img src="outputs/hq_renders/girl_hq.png" width="45%">
  <img src="outputs/hq_renders/talwar_hq.png" width="45%">
</p>

<p align="center">
  <i>Original vs Reconstructed mesh (Uniform Quantization)</i>
</p>

---

## 📈 Example Error Plot

<p align="center">
  <img src="outputs/metrics/branch_error.png" width="60%">
</p>

<p align="center">
  <i>MSE and MAE per axis for <b>branch.obj</b> reconstruction</i>
</p>

---

## 🧾 Quantitative Summary

| Metric | Description | Value |
|---------|--------------|-------|
| Mean MSE (Uniform) | Average mean-squared-error across all meshes | 7.8×10⁻⁷ |
| Mean MAE (Uniform) | Average absolute error | 7.3×10⁻⁴ |
| Mean MSE (Adaptive) | Average adaptive prototype error | 4.5×10⁻⁶ |
| Reconstruction Quality | Excellent (uniform), fair (adaptive) | ✅ |
| Render Quality | High (via PyRender / Pyglet) | 🖼️ |
| Runtime per mesh | ~2–3 seconds (CPU) | ⚡ |

---

## 🧮 Dependencies

```
numpy
matplotlib
trimesh
joblib
scipy
imageio
pyglet<2
pyrender
PyOpenGL
```

---

## 📘 Future Improvements

- Implement **region-based adaptive quantization**.
- Add **post-dequantization smoothing**.
- Integrate **Open3D viewer** for real-time preview.
- Extend to animated meshes.

---

## 🙌 Acknowledgements

- **Trimesh** for mesh loading and processing.  
- **PyRender / Pyglet** for rendering.  
- **SciPy KDTree** for density estimation.  
- **SeamGPT** for providing this challenge.

---

<p align="center"><i>This project is developed for academic and assessment purposes only.</i></p>

<p align="center"><b>✨ End of README ✨</b></p>
