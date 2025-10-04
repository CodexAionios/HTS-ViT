from __future__ import annotations
import numpy as np

# --- Convolution on (approx) hypersphere via cosine kernel ---

def geodesic_convolution(T: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Geodesic-style convolution using cosine distances as geodesics on S^{d-1}.
    T: (N, d)
    """
    X = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    cos = X @ X.T
    theta = np.arccos(np.clip(cos, -1.0, 1.0))
    K = np.exp(-(theta**2) / (2 * sigma**2))
    K /= (K.sum(axis=1, keepdims=True) + 1e-8)
    return K @ T

# --- Dissemination & its inverse operations ---

def dissemination(T: np.ndarray, hyperedges: list[list[int]], weight: float = 0.5) -> np.ndarray:
    """Spread tensor info by averaging over hyperedges (simple broadcast)."""
    X = T.copy()
    for e in hyperedges:
        avg = T[e, :].mean(axis=0, keepdims=True)
        X[e, :] = (1 - weight) * T[e, :] + weight * avg
    return X


def deconvolution(T: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Sharpen signals (unsharp masking style) as a deconvolution proxy."""
    blurred = geodesic_convolution(T)
    return T + alpha * (T - blurred)


def insemination(T: np.ndarray, inserts: dict[int, np.ndarray]) -> np.ndarray:
    """Targeted insertion of token values at specified indices."""
    X = T.copy()
    for idx, vec in inserts.items():
        X[idx, :] = vec
    return X
