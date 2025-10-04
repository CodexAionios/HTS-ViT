from __future__ import annotations
import numpy as np
from .hypergraph import Hypergraph, hypergraph_laplacian


def hts_attention(T: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                  G: Hypergraph | None = None, laplacian_strength: float = 0.0) -> np.ndarray:
    """Scaled dot-product attention with optional hypergraph geometric prior.

    T: (N, d) tokens; W* : (d, d)
    Returns: (N, d)
    """
    Q = T @ Wq
    K = T @ Wk
    V = T @ Wv
    dk = K.shape[1]
    logits = (Q @ K.T) / np.sqrt(dk)
    if G is not None and laplacian_strength > 0:
        L = hypergraph_laplacian(G, variant="zhou")  # (N,N)
        # Encourage attention along hypergraph relations; subtract Laplacian energy
        logits = logits - laplacian_strength * L
    # softmax row-wise
    logits = logits - logits.max(axis=1, keepdims=True)
    A = np.exp(logits)
    A /= (A.sum(axis=1, keepdims=True) + 1e-8)
    return A @ V


def hts_block(T: np.ndarray, G: Hypergraph | None = None, laplacian_strength: float = 0.2,
              seed: int | None = 0) -> np.ndarray:
    """One HTS‑ViT style block: attention over tensorfield with hypergraph prior + residual.
    Weights are randomly initialised for didactic demo.
    """
    rng = np.random.default_rng(seed)
    d = T.shape[1]
    Wq = rng.normal(scale=0.5, size=(d, d))
    Wk = rng.normal(scale=0.5, size=(d, d))
    Wv = rng.normal(scale=0.5, size=(d, d))
    att = hts_attention(T, Wq, Wk, Wv, G, laplacian_strength)
    return T + att  # simple residual
