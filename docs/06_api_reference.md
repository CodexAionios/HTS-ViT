# API Reference (short)

- `Hypergraph(n, hyperedges, weights=None)`
- `hypergraph_laplacian(G, variant='zhou') -> np.ndarray`
- `hts_attention(T, Wq, Wk, Wv, G=None, laplacian_strength=0.0)`
- `hts_block(T, G=None, laplacian_strength=0.2)`
- `datatropic_step(T, G=None, strength=None)`
- `geodesic_convolution(T, sigma=0.8)`
- `dissemination(T, hyperedges, weight=0.5)`
- `deconvolution(T, alpha=0.2)`
- `insemination(T, inserts)`
- `sobolev_inner(F,G,order=1)`
- `project_to_manifold(T, P=None)`
- `riemann_dot(X,Y,G=None)`
- `spherical_radius(Pi,Pj,metric=None)`
- `hts_vit_step(T, G=None)`
- `HTSRenderer.render_interactive(E_t=0.0)`
