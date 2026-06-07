"""Matplotlib figure builders for the GUI.

These return Figure objects (rendered with the headless Agg backend) so the
Streamlit layer can display them via ``st.pyplot``. They deliberately depend
only on matplotlib / networkx (not Streamlit), so they are unit-testable on
their own. The topology / ΔV styling mirrors ``fault/reporter.py``.
"""
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # Headless; safe under Streamlit's server process.
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def topology_figure(circuit, faulty_nodes: Optional[Sequence[int]] = None):
    """Circuit topology graph. Red = diagnosed faulty, Blue = accessible (ADC),
    Grey = internal/inaccessible."""
    faulty_nodes = set(faulty_nodes or [])
    fig, ax = plt.subplots(figsize=(6, 4))

    G = nx.Graph()
    for node in circuit.config.nodes:
        G.add_node(node)
    for el in circuit.config.elements:
        G.add_edge(el.n1, el.n2, label=el.name)

    pos = nx.spring_layout(G, seed=42)  # Fixed seed -> stable layout.
    accessible_set = set(circuit.config.accessible)

    node_colors = []
    for node in G.nodes():
        if node in faulty_nodes:
            node_colors.append("red")
        elif node in accessible_set:
            node_colors.append("skyblue")
        else:
            node_colors.append("lightgray")

    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
            node_size=600, font_size=10, font_weight="bold", edge_color="gray")
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax,
        edge_labels={(u, v): d["label"] for u, v, d in G.edges(data=True)},
        font_size=8,
    )
    ax.set_title("Circuit Topology (Red: Faulty, Blue: ADC)")
    fig.tight_layout()
    return fig


def _accessible_labels(circuit) -> List[str]:
    nodes = [n for n in circuit.config.accessible if n != circuit.reference_node]
    return [str(n) for n in nodes]


def delta_v_figure(circuit, delta_v_m):
    """Bar chart of the voltage deviation at accessible nodes. For AC (complex
    ΔV) shows magnitude and phase in two stacked panels."""
    labels = _accessible_labels(circuit)
    dv = np.asarray(delta_v_m)

    if np.iscomplexobj(dv):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)
        ax1.bar(labels, np.abs(dv), color="orange")
        ax1.set_title("Voltage Deviation Magnitude |Delta Vm|")
        ax1.set_ylabel("|Delta V| [V]")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        ax2.bar(labels, np.angle(dv, deg=True), color="seagreen")
        ax2.set_title("Voltage Deviation Phase (Delta Vm)")
        ax2.set_xlabel("Node")
        ax2.set_ylabel("Phase [deg]")
        ax2.grid(axis="y", linestyle="--", alpha=0.7)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, dv, color="orange")
        ax.set_title("Voltage Deviation at Accessible Nodes (Delta Vm)")
        ax.set_xlabel("Node")
        ax.set_ylabel("Voltage Deviation [V]")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        fig.tight_layout()
    return fig
