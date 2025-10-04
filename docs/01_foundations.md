# Foundations: Tensorfield & HTS

A tensorfield \(\mathcal{T} = [T_1,\dots,T_N]\) lives on hypergraph nodes (image patches, agents, etc.).

**Hypergraph Laplacian (Zhou):**
\[L = I - D_v^{-1/2} H W D_e^{-1} H^\top D_v^{-1/2}\]

**HTS Attention:**
\[ V^{\wedge}_{\mathrm{Att}}(T) = \mathrm{softmax}\big(\tfrac{TW_Q (TW_K)^\top}{\sqrt d}\big)\, TW_V \]

**Evolution (datatropic PDE, discrete step):**
\[\Delta T = \lambda_1\,\nabla_{\text{Att}} + \lambda_2\,\rho + \lambda_3\,F_{\text{Edge}} + \dots\]
We implement practical proxies for each term in `datatropisms.py`.
