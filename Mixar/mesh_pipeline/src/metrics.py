# src/metrics.py
"""
Metrics utilities for mesh reconstruction quality.

Functions:
- compute_errors(original, reconstructed)
- save_error_metrics(errors, out_json_path)
- plot_error_per_axis(errors, out_path)
"""

import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # for non-interactive environments
import matplotlib.pyplot as plt


def compute_errors(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """
    Compute MSE and MAE per axis and overall.
    Args:
        original: (N,3) array of original vertices
        reconstructed: (N,3) array of reconstructed vertices
    Returns:
        dict with mse_per_axis, mae_per_axis, mse, mae
    """
    diff = reconstructed - original
    mse_per_axis = np.mean(diff ** 2, axis=0)
    mae_per_axis = np.mean(np.abs(diff), axis=0)
    mse = float(np.mean(mse_per_axis))
    mae = float(np.mean(mae_per_axis))
    return {
        "mse_per_axis": mse_per_axis.tolist(),
        "mae_per_axis": mae_per_axis.tolist(),
        "mse": mse,
        "mae": mae
    }


def save_error_metrics(errors: dict, out_json_path: str):
    """
    Save error metrics as JSON.
    """
    p = Path(out_json_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)


def plot_error_per_axis(errors: dict, out_path: str, title: str = ""):
    """
    Plot MSE and MAE per axis as grouped bar chart.
    """
    mse_axes = np.array(errors["mse_per_axis"])
    mae_axes = np.array(errors["mae_per_axis"])
    labels = ['X', 'Y', 'Z']
    x = np.arange(3)
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, mse_axes, width, label='MSE')
    ax.bar(x + width/2, mae_axes, width, label='MAE')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title or "Reconstruction Error per Axis")
    ax.legend()
    plt.tight_layout()

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200)
    plt.close(fig)
