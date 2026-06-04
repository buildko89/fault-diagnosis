import os
import datetime
import matplotlib
matplotlib.use('Agg') # Headless mode
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any, List
import numpy as np

def generate_markdown_report(
    circuit, 
    delta_v_m: np.ndarray, 
    result_omp: Dict[str, Any], 
    result_legacy: Dict[str, Any], 
    output_dir: str = './report'
) -> str:
    """
    Generates a markdown report with topology and voltage deviation plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate topology.png
    plt.figure(figsize=(6, 4))
    G = nx.Graph()
    for node in circuit.config.nodes:
        G.add_node(node)
    for el in circuit.config.elements:
        G.add_edge(el.n1, el.n2, label=el.name)
        
    # Fixed seed for consistent layout
    pos = nx.spring_layout(G, seed=42)
    
    node_colors = []
    faulty_nodes = []
    if result_omp['result']['best']:
        faulty_nodes = result_omp['result']['best']['support']
        
    accessible_set = set(circuit.config.accessible)
    for node in G.nodes():
        if node in faulty_nodes:
            node_colors.append('red') # Faulty nodes
        elif node in accessible_set:
            node_colors.append('skyblue') # Accessible (ADC) nodes
        else:
            node_colors.append('lightgray') # Normal inaccessible nodes
            
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=600, font_size=10, font_weight='bold', edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['label'] for u, v, d in G.edges(data=True)}, font_size=8)
    
    plt.title("Circuit Topology (Red: Faulty, Blue: ADC)")
    topology_path = os.path.join(output_dir, 'topology.png')
    plt.tight_layout()
    plt.savefig(topology_path)
    plt.close()
    
    # 2. Generate delta_v.png
    # delta_v_m is ordered/sized by get_accessible_indices() which excludes the
    # reference node; derive the x labels from the same source so the bar count
    # always matches the data (config.accessible may include the reference node).
    nodes = [n for n in circuit.config.accessible if n != circuit.reference_node]
    labels = [str(n) for n in nodes]
    dv = np.asarray(delta_v_m)
    delta_v_path = os.path.join(output_dir, 'delta_v.png')

    if np.iscomplexobj(dv):
        # AC: show magnitude and phase of the complex voltage deviation.
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)
        ax1.bar(labels, np.abs(dv), color='orange')
        ax1.set_title("Voltage Deviation Magnitude |Delta Vm|")
        ax1.set_ylabel("|Delta V| [V]")
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.bar(labels, np.angle(dv, deg=True), color='seagreen')
        ax2.set_title("Voltage Deviation Phase (Delta Vm)")
        ax2.set_xlabel("Node")
        ax2.set_ylabel("Phase [deg]")
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
    else:
        plt.figure(figsize=(6, 4))
        plt.bar(labels, dv, color='orange')
        plt.title("Voltage Deviation at Accessible Nodes (Delta Vm)")
        plt.xlabel("Node")
        plt.ylabel("Voltage Deviation [V]")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

    plt.savefig(delta_v_path)
    plt.close()
    
    # 3. Generate Markdown Report
    best_omp = result_omp['result'].get('best', {})
    best_leg = result_legacy['result'].get('best', {})
    
    omp_nodes = best_omp.get('support', [])
    omp_err = best_omp.get('residual_norm', 0.0)
    leg_nodes = best_leg.get('support', [])
    leg_err = best_leg.get('residual_norm', 0.0)
    
    md_content = f"""# Analog Circuit Fault Diagnosis Report

**Date Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Comparison

| Method | Execution Time (s) | Diagnosed Faulty Nodes | Projection Error (Residual) |
|---|---|---|---|
| OMP (Sparse Approximation) | {result_omp['time']:.6f} | {omp_nodes} | {omp_err:.6e} |
| Exhaustive (Dense Search) | {result_legacy['time']:.6f} | {leg_nodes} | {leg_err:.6e} |

## Visualization

### Topological Fault Location
![Topology](topology.png)

*Grey: Internal Nodes / Blue: Accessible (ADC) Nodes / Red: Diagnosed Faulty Nodes (OMP)*

### Voltage Deviations
![Delta V](delta_v.png)

*Voltage deviations at accessible nodes under the first current excitation.*
"""
    
    md_path = os.path.join(output_dir, 'diagnosis_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    return md_content
