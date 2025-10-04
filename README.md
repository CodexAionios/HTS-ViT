# Tensorfield-HTS

Hypergraph Tensor Supersystems (HTS) + HTS‑ViT + CDMNs + Datatropisms.

**Why**: Move beyond vector‑centric pipelines. Embed tensors on hypergraphs/manifolds; let attention and flows respect geometry & topology. Teach linear algebra geometrically/computationally for real systems (AI/ML/LLMs).

## Key Concepts (short)
- **Tensorfield**: dynamic tensor field \(\mathcal{T}\) on nodes/patches of a hypergraph \(H=(V,E)\) and (optionally) a manifold.
- **Hypergraph Laplacian**: spatial/relational prior \(\Delta_H\) coupling patches.
- **Datatropisms**: forces guiding evolution: gradient‑, entrop‑, correlat‑, anisotrop‑isms.
- **Flows**: convolution ⇄ deconvolution; dissemination ⇄ insemination.
- **Riemann Manifold Puncture**: bridge from hypersphere representations to reality via metric projections & Sobolev inner products.
- **HTS‑ViT**: self‑attention operating inside the tensorfield, modulated by \(H\) and tropisms.

## Install
```bash
pip install -e .
```

## Quickstart

```python
from tensorfield import Hypergraph, hts_block, datatropic_step
import numpy as np

# Toy tokens on 5 patches
T = np.random.randn(5, 32)
H = Hypergraph(n=5, hyperedges=[[0,1,2],[2,3,4]])

for _ in range(3):
    T = hts_block(T, H)              # HTS attention + hypergraph prior
    T = datatropic_step(T, H)        # gradient/entropy/correlation/anisotropy forces
print(T.shape)  # (5, 32)
```

## Papers/Notes

* See `docs/` for foundations and API.
* `examples/` for runnable demos.
```
