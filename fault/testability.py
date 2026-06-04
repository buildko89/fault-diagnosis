import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from typing import List, Tuple, Dict
from .circuit import Circuit

def check_k_node_testability(circuit: Circuit, k: int) -> Tuple[bool, Dict[int, int]]:
    """
    Checks k-node fault testability.
    Any inaccessible node must have >= k+1 vertex-disjoint paths to accessible nodes (including ground).
    """
    inaccessible = circuit.get_inaccessible_nodes()
    accessible_all = set(circuit.config.accessible) | {circuit.reference_node}
    
    # Build unique edges (global node indices, undirected)
    edges = set()
    for el in circuit.config.elements:
        u, v = min(el.n1, el.n2), max(el.n1, el.n2)
        edges.add((u, v))
        
    testable = True
    node_connectivities = {}
    
    num_nodes = circuit.num_nodes
    # Node splitting for vertex capacity: Node i -> (2*i, 2*i + 1)
    # Super-sink for all accessible nodes
    sink = 2 * num_nodes
    size = sink + 1
    
    inf_cap = num_nodes + 1 # Sufficiently large capacity for vertex disjointness

    for src_node in inaccessible:
        src_idx = circuit.node_to_idx[src_node]
        
        rows, cols, caps = [], [], []
        
        # Vertex capacities (node splitting)
        for i in range(num_nodes):
            u_in = 2 * i
            u_out = 2 * i + 1
            rows.append(u_in)
            cols.append(u_out)
            
            node_val = circuit.idx_to_node[i]
            if node_val in accessible_all:
                caps.append(inf_cap)
            else:
                caps.append(1) # Max 1 path through this vertex
                
        # Edge capacities
        for u_node, v_node in edges:
            u_idx = circuit.node_to_idx[u_node]
            v_idx = circuit.node_to_idx[v_node]
            
            # u_out -> v_in
            rows.append(2 * u_idx + 1)
            cols.append(2 * v_idx)
            caps.append(1)
            
            # v_out -> u_in
            rows.append(2 * v_idx + 1)
            cols.append(2 * u_idx)
            caps.append(1)
            
        # Connect all accessible nodes to super-sink
        for acc_node in accessible_all:
            acc_idx = circuit.node_to_idx[acc_node]
            rows.append(2 * acc_idx + 1)
            cols.append(sink)
            caps.append(inf_cap)
            
        flow_graph = csr_matrix((caps, (rows, cols)), shape=(size, size))
        flow_source = 2 * src_idx + 1 # Start from the 'out' side of the source node
        
        res = maximum_flow(flow_graph, flow_source, sink)
        conn = int(round(res.flow_value))
        node_connectivities[src_node] = conn
        
        if conn < k + 1:
            testable = False
            
    return testable, node_connectivities
