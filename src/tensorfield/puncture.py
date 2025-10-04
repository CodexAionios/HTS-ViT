from __future__ import annotations
import numpy as np

# --- Sobolev-like inner product (discrete proxy) ---

def sobolev_inner(F: np.ndarray, G: np.ndarray, order: int = 1) -> float:
    """Discrete Sobolev inner product W^{k,2} on sequences of tokens (proxy).
    Treat rows as samples; finite-difference along sample index.
    """
    assert F.shape == G.shape
    base = float(np.sum(F * G))
    acc = base
    dF, dG = F.copy(), G.copy()
    for _ in range(order):
        dF = np.diff(dF, axis=0, prepend=dF[:1])
        dG = np.diff(dG, axis=0, prepend=dG[:1])
        acc += float(np.sum(dF * dG))
    return acc

# --- Riemann manifold projection & metric dot ---

def project_to_manifold(T: np.ndarray, P: np.ndarray | None = None) -> np.ndarray:
    """Linear projection to a tangent chart: Y = T @ P (default: identity)."""
    if P is None:
        return T
    return T @ P


def riemann_dot(X: np.ndarray, Y: np.ndarray, G: np.ndarray | None = None) -> float:
    """Dot using metric tensor G (positive definite)."""
    if G is None:
        return float(np.sum(X * Y))
    return float(np.sum((X @ G) * Y))


def spherical_radius(Pi: np.ndarray, Pj: np.ndarray, metric: np.ndarray | None = None) -> float:
    """Confidence radius r = sqrt(<Pi,Pi> - <Pi,Pj>)."""
    a = riemann_dot(Pi, Pi, metric)
    b = riemann_dot(Pi, Pj, metric)
    return float(np.sqrt(max(0.0, a - b)))
