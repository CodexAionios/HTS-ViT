import numpy as np
from tensorfield import Hypergraph
from tensorfield.hts_vit import hts_vit_step

N, d = 8, 32
T = np.random.randn(N, d)
G = Hypergraph(n=N, hyperedges=[[0,1,2],[2,3,4],[4,5,6,7]])

for _ in range(4):
    T = hts_vit_step(T, G)
print(T.mean(), T.std())
