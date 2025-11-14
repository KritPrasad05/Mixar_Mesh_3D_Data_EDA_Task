# src/transforms.py
"""
Normalizers and Quantizer for mesh vertices.

Classes:
- MinMaxNormalizer
- UnitSphereNormalizer
- Quantizer
"""

from typing import Optional, Dict, Any
import numpy as np


class MinMaxNormalizer:
    """
    Min-Max normalization to [0,1] per axis.
    Stores v_min, v_max, and handles zero-range axes.
    """
    def __init__(self):
        self.v_min: Optional[np.ndarray] = None
        self.v_max: Optional[np.ndarray] = None
        self.range_: Optional[np.ndarray] = None

    def fit(self, vertices: np.ndarray) -> "MinMaxNormalizer":
        v = np.asarray(vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError("vertices must be (N,3)")
        self.v_min = v.min(axis=0)
        self.v_max = v.max(axis=0)
        self.range_ = self.v_max - self.v_min
        # avoid zero division
        zero_mask = self.range_ == 0.0
        if np.any(zero_mask):
            # set zero ranges to 1.0 so transform is identity on that axis
            self.range_[zero_mask] = 1.0
        return self

    def transform(self, vertices: np.ndarray) -> np.ndarray:
        v = np.asarray(vertices, dtype=float)
        return (v - self.v_min) / self.range_

    def inverse_transform(self, norm_vertices: np.ndarray) -> np.ndarray:
        nv = np.asarray(norm_vertices, dtype=float)
        return nv * self.range_ + self.v_min

    def get_metadata(self) -> Dict[str, Any]:
        return {"method": "minmax", "v_min": self.v_min, "v_max": self.v_max}


class UnitSphereNormalizer:
    """
    Center mesh at mean and scale so max distance from center = 1 (unit sphere).
    """
    def __init__(self):
        self.center_: Optional[np.ndarray] = None
        self.max_dist_: Optional[float] = None

    def fit(self, vertices: np.ndarray) -> "UnitSphereNormalizer":
        v = np.asarray(vertices, dtype=float)
        self.center_ = v.mean(axis=0)
        dists = np.linalg.norm(v - self.center_, axis=1)
        self.max_dist_ = float(np.max(dists))
        if self.max_dist_ == 0.0:
            self.max_dist_ = 1.0
        return self

    def transform(self, vertices: np.ndarray) -> np.ndarray:
        v = np.asarray(vertices, dtype=float)
        return (v - self.center_) / self.max_dist_

    def inverse_transform(self, norm_vertices: np.ndarray) -> np.ndarray:
        nv = np.asarray(norm_vertices, dtype=float)
        return nv * self.max_dist_ + self.center_

    def get_metadata(self) -> Dict[str, Any]:
        return {"method": "unitsphere", "center": self.center_, "max_dist": self.max_dist_}


class Quantizer:
    """
    Uniform quantizer for normalized data in [0,1].
    n_bins: integer number of bins (e.g., 1024)
    Note: expects inputs in [0,1] (if using UnitSphere and values can be negative, you
    should map from [-1,1] to [0,1] externally or use minmax).
    """
    def __init__(self, n_bins: int = 1024, clamp: bool = True):
        self.n_bins = int(n_bins)
        self.clamp = bool(clamp)
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2")

    def transform(self, norm_vertices: np.ndarray) -> np.ndarray:
        """
        Quantize normalized vertices to integer bins [0, n_bins-1]
        Returns int array shape (N,3)
        """
        nv = np.asarray(norm_vertices, dtype=float)
        # If values may be outside [0,1], clamp or wrap as required
        if self.clamp:
            nv = np.clip(nv, 0.0, 1.0)
        q = np.floor(nv * (self.n_bins - 1)).astype(np.int32)
        return q

    def inverse_transform(self, quantized: np.ndarray) -> np.ndarray:
        """
        Convert integer bins back to normalized floats in [0,1]
        """
        q = np.asarray(quantized, dtype=float)
        return q / (self.n_bins - 1)

    def get_metadata(self) -> Dict[str, Any]:
        return {"n_bins": self.n_bins}
