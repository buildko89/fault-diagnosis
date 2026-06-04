import numpy as np
from typing import List, Dict, Optional, Tuple
from .schema import CircuitConfig, Element
import scipy.sparse as sp


def element_admittance(el_type: str, value: float, omega: Optional[float]):
    """
    Complex admittance of an element at angular frequency ``omega`` [rad/s].

        R -> value            (conductance G; frequency-independent, real)
        C -> j*omega*value    (value = capacitance)
        L -> 1/(j*omega*value)(value = inductance)

    ``omega=None`` denotes DC and is only valid for resistors; reactive types
    raise (this is normally prevented earlier by schema validation).
    """
    if el_type == 'R':
        return value
    if omega is None:
        raise ValueError(
            f"Reactive element type '{el_type}' requires an angular frequency (omega)."
        )
    if el_type == 'C':
        return 1j * omega * value
    if el_type == 'L':
        return 1.0 / (1j * omega * value)
    raise ValueError(f"Unsupported element type '{el_type}'.")


class Circuit:
    def __init__(self, config: CircuitConfig):
        self.config = config
        self.num_nodes = len(config.nodes)
        self.reference_node = config.reference
        
        # Map node values to 0-based indices for internal matrices
        # Index 0 is reserved for the ground (reference) if possible, 
        # but the plan says "reduced nodal admittance matrix".
        # We'll map nodes to indices 0..n-1, and then exclude the reference node index.
        self.node_to_idx = {node: i for i, node in enumerate(config.nodes)}
        self.idx_to_node = {i: node for node, i in self.node_to_idx.items()}
        
        self.ref_idx = self.node_to_idx[self.reference_node]
        
        # Indices of non-reference nodes (free nodes)
        self.free_indices = [i for i in range(self.num_nodes) if i != self.ref_idx]
        self.num_free_nodes = len(self.free_indices)
        
        # Mapping from global node index to reduced matrix row/column index
        self.global_to_reduced = {g_idx: r_idx for r_idx, g_idx in enumerate(self.free_indices)}

    def build_matrices(self, faulty_elements: Optional[Dict[str, float]] = None,
                       rng: Optional[np.random.Generator] = None,
                       tol_percent: float = 0.0,
                       omega: Optional[float] = None) -> Tuple[sp.spmatrix, sp.spmatrix, sp.spmatrix]:
        """
        Builds A (reduced incidence), Yb (branch admittance), and Y (reduced nodal admittance).
        A: (num_free_nodes, num_elements)
        Yb: (num_elements, num_elements)
        Y: (num_free_nodes, num_free_nodes)

        omega : angular frequency [rad/s]. None -> DC (resistors only); the branch
        admittances are then real (backward-compatible). When omega is given, the
        matrices are complex (Yb = element_admittance(type, value, omega)).

        Tolerance/fault perturbations are applied to the element *value* (the
        physical component value, i.e. G/C/L) before the admittance is computed,
        so faulty_elements[name] is interpreted in the same units as Element.value.
        """
        num_elements = len(self.config.elements)
        use_complex = omega is not None
        dtype = complex if use_complex else float

        A = sp.lil_matrix((self.num_free_nodes, num_elements))
        Yb = sp.lil_matrix((num_elements, num_elements), dtype=dtype)

        for e_idx, el in enumerate(self.config.elements):
            value_nominal = el.value
            value = value_nominal

            is_faulty = False
            if faulty_elements and el.name in faulty_elements:
                value = faulty_elements[el.name]
                is_faulty = True

            if not is_faulty and tol_percent > 0.0 and rng is not None:
                std_dev = (tol_percent / 100.0) * value_nominal
                value = value_nominal + rng.normal(0, std_dev)
                value = max(value, 1e-9) # Ensure positive

            Yb[e_idx, e_idx] = element_admittance(el.type, value, omega)

            # Flow el.n1 -> el.n2
            idx1 = self.node_to_idx[el.n1]
            idx2 = self.node_to_idx[el.n2]

            if idx1 in self.global_to_reduced:
                A[self.global_to_reduced[idx1], e_idx] = 1.0
            if idx2 in self.global_to_reduced:
                A[self.global_to_reduced[idx2], e_idx] = -1.0

        A = A.tocsr()
        Yb = Yb.tocsr()
        Y = A @ Yb @ A.T
        return A, Yb, Y

    def get_accessible_indices(self) -> List[int]:
        """Returns indices in the reduced matrix (0 to num_free_nodes-1)"""
        return [self.global_to_reduced[self.node_to_idx[n]] 
                for n in self.config.accessible if n != self.reference_node]

    def get_inaccessible_nodes(self) -> List[int]:
        acc_set = set(self.config.accessible) | {self.reference_node}
        return [n for n in self.config.nodes if n not in acc_set]
