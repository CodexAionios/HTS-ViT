from __future__ import annotations
import numpy as np
from .hypergraph import Hypergraph, hypergraph_laplacian


def _row_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def gradientropism(T: np.ndarray, A: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Pull tokens toward high-attention centroids (discrete attention-gradient proxy).
    A: (N,N) attention matrix (row-stochastic). Returns ΔT.
    """
    centroid = A @ T
    return strength * (centroid - T)


def entropism(T: np.ndarray, strength: float = 0.05) -> np.ndarray:
    """Spread features to increase diversity: push rows apart (repulsion). Returns ΔT."""
    X = _row_norm(T)
    S = X @ X.T  # cos sim
    rep = (S * 1.0 - np.eye(T.shape[0])) @ T  # push away from similar
    return -strength * rep / T.shape[0]


def correlatropism(T: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """Align with principal correlation directions (PCA-like). Returns ΔT."""
    T0 = T - T.mean(axis=0, keepdims=True)
    C = (T0.T @ T0) / max(1, T.shape[0]-1)
    # Project onto top-k eigens (k = min(4, d))
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    V_top = V[:, idx[: min(4, V.shape[1])]]
    proj = (T0 @ V_top) @ V_top.T
    return strength * (proj - T0)


def anisotropism(T: np.ndarray, direction: np.ndarray | None = None, strength: float = 0.1) -> np.ndarray:
    """Directional drift along an external bias vector in feature space."""
    if direction is None:
        direction = np.ones(T.shape[1])
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return strength * (T @ np.outer(direction, direction) - T * 0.0)


def datatropic_step(T: np.ndarray, G: Hypergraph | None = None, strength: dict | None = None) -> np.ndarray:
    """Apply combined datatropic evolution once.
    strength: optional dict of coefficients.
    """
    st = {"grad": 0.6, "entr": 0.03, "corr": 0.15, "anis": 0.1}
    if strength: st.update(strength)
    # lightweight attention proxy for gradientropism
    QK = T @ T.T / np.sqrt(T.shape[1])
    QK = QK - QK.max(axis=1, keepdims=True)
    A = np.exp(QK); A /= (A.sum(axis=1, keepdims=True) + 1e-8)
    dT = (
        st["grad"] * gradientropism(T, A, strength=1.0)
        + st["entr"] * entropism(T)
        + st["corr"] * correlatropism(T)
        + st["anis"] * anisotropism(T)
    )
    # Optional geometric smoothing along hypergraph
    if G is not None:
        L = hypergraph_laplacian(G)
        dT = dT - 0.05 * (L @ T)
    return T + dT
