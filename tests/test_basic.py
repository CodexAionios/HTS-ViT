import numpy as np
from tensorfield import Hypergraph, hts_block, datatropic_step, geodesic_convolution

def test_pipeline_runs():
    T = np.random.randn(5, 8)
    G = Hypergraph(5, [[0,1,2],[2,3,4]])
    T = hts_block(T, G)
    T = geodesic_convolution(T)
    T = datatropic_step(T, G)
    assert T.shape == (5, 8)
