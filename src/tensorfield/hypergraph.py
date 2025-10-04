from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class Hypergraph:
    n: int
    hyperedges: list[list[int]]
    weights: list[float] | None = None

    def incidence(self) -> np.ndarray:
        m = len(self.hyperedges)
        H = np.zeros((self.n, m), dtype=float)
        for j, e in enumerate(self.hyperedges):
            for v in e:
                H[v, j] = 1.0
        return H

    def weight_matrix(self) -> np.ndarray:
        m = len(self.hyperedges)
        w = np.ones(m) if self.weights is None else np.asarray(self.weights, dtype=float)
        return np.diag(w)


def hypergraph_laplacian(G: Hypergraph, variant: str = "zhou") -> np.ndarray:
    """Return a hypergraph Laplacian.

    * zhou: L = I - Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}
    * simple: Δ_H = Dv^{-1} H W H^T Dv^{-1}
    """
    H = G.incidence()
    W = G.weight_matrix()
    dv = H @ W @ np.ones((H.shape[1], 1))  # vertex degree
    Dv = np.diagflat(dv.ravel() + 1e-8)
    if variant == "zhou":
        De = np.diag((H.sum(axis=0) + 1e-8))  # hyperedge size diag
        Dv_mhalf = np.linalg.inv(np.sqrt(Dv))
        De_inv = np.linalg.inv(De)
        I = np.eye(G.n)
        L = I - Dv_mhalf @ H @ W @ De_inv @ H.T @ Dv_mhalf
        return L
    else:
        Dv_inv = np.linalg.inv(Dv)
        return Dv_inv @ H @ W @ H.T @ Dv_inv
