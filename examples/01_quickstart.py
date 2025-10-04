import numpy as np
from tensorfield import Hypergraph, hts_block, datatropic_step

N, d = 6, 16
T = np.random.randn(N, d)
G = Hypergraph(n=N, hyperedges=[[0,1,2],[2,3,4,5]])

for it in range(5):
    T = hts_block(T, G)
    T = datatropic_step(T, G)
print("Final shape:", T.shape)
