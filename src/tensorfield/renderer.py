# Adapted from the interactive HTSRenderer prototype (Plotly + NetworkX).
import numpy as np
import plotly.graph_objects as go
import networkx as nx

class HTSRenderer:
    def __init__(self, alpha=(2.0,0.5,0.3), beta=(1.5,0.7), theta=1.0):
        self.a0, self.a1, self.a2 = alpha
        self.b0, self.b1 = beta
        self.theta = theta
        self.nodos = ["Qd", "Qs", "P", "E"]
        self.pos = {"Qd": (0, 1, 0), "Qs": (0, -1, 0), "P": (1.2, 0, 0), "E": (-1.2, 0, 0)}
        self._build_hypergraph()

    def equilibrio_log_price(self, E_t):
        return (self.a0 - self.b0 + self.a2 * E_t) / (self.a1 + self.b1)

    def estado_mercado(self, E_t):
        lnP = self.equilibrio_log_price(E_t)
        lnQd = self.a0 - self.a1 * lnP + self.a2 * E_t
        lnQs = self.b0 + self.b1 * lnP
        return np.array([lnQd, lnQs, lnP])

    def operador_V(self, psi, E_t):
        return psi + self.theta * (self.estado_mercado(E_t) - psi)

    def _build_hypergraph(self):
        G = nx.DiGraph()
        G.add_nodes_from(self.nodos)
        G.add_edge("E", "Qd", weight=self.a2)
        G.add_edge("P", "Qd", weight=-self.a1)
        G.add_edge("P", "Qs", weight=self.b1)
        self.hypergraph = G

    def render_interactive(self, E_t=0.0):
        psi0 = self.estado_mercado(0.0)
        psi1 = self.operador_V(psi0, E_t)
        fig = go.Figure()
        for node, (x, y, z) in self.pos.items():
            fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers+text',
                                       marker=dict(size=10), text=[node], textposition="top center"))
        for u, v, d in self.hypergraph.edges(data=True):
            x0, y0, z0 = self.pos[u]; x1, y1, z1 = self.pos[v]
            fig.add_trace(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1], mode='lines',
                                       line=dict(width=2, color='gray'), name=f"{u}→{v} ({d['weight']:.2f})"))
        for i, var in enumerate(["Qd", "Qs", "P"]):
            _, y, _ = self.pos[var]
            z0, z1 = psi0[i], psi1[i]
            fig.add_trace(go.Scatter3d(x=[1.5, 1.5], y=[y, y], z=[z0, z1], mode='lines+markers',
                                       marker=dict(size=4), line=dict(color='red', width=4), name=f'{var}'))
        fig.update_layout(title=f"HTS Flow – Externalidad E_t = {E_t:.2f}",
                          scene=dict(xaxis_title='Estructura', yaxis_title='Variable', zaxis_title='Log-valor'),
                          margin=dict(l=0, r=0, t=50, b=0), height=600)
        return fig
