from __future__ import annotations
import numpy as np
from .attention import hts_attention
from .hypergraph import Hypergraph
from .cdmn import geodesic_convolution, dissemination, deconvolution, insemination
from .datatropisms import datatropic_step


def hts_vit_step(T: np.ndarray, G: Hypergraph | None = None, seed: int | None = 0) -> np.ndarray:
    """One composite HTS‑ViT step: attention ⟶ geodesic conv ⟶ dissemination ⟶ tropisms."""
    T1 = hts_attention(T, *(np.eye(T.shape[1]),)*3, G, laplacian_strength=0.2)
    T2 = geodesic_convolution(T1)
    if G is not None:
        T3 = dissemination(T2, G.hyperedges, weight=0.4)
    else:
        T3 = T2
    T4 = datatropic_step(T3, G)
    return T4
