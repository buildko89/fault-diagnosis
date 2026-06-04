import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import List, Dict, Optional, Tuple
from .circuit import AnalogCircuit

def solve_response(Y: sp.spmatrix, J: np.ndarray) -> np.ndarray:
    """Solves Y * V = J for V using a stable sparse solver."""
    return spla.spsolve(Y, J)

def calculate_delta_v(circuit: AnalogCircuit, excitations: List[np.ndarray], 
                      faulty_elements: Optional[Dict[str, float]] = None, 
                      rng: Optional[np.random.Generator] = None, 
                      tol_percent: float = 0.0) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Returns:
        delta_v_ms: List of deviation vectors at accessible nodes.
        Z_mn: Transfer impedance matrix (nominal) for accessible nodes.
    """
    # Nominal matrices
    A_nom, Yb_nom, Y_nom = circuit.build_matrices(faulty_elements=None)
    
    # Faulty matrices
    A_f, Yb_f, Y_f = circuit.build_matrices(faulty_elements=faulty_elements, rng=rng, tol_percent=tol_percent)
    
    acc_indices = circuit.get_accessible_indices()
    
    # Pre-calculate Z_mn = [inv(Y_nom)]_accessible_rows
    # To avoid full inversion, we can solve for unit injections at accessible nodes if m < n,
    # but the formula is delta_V = Z * delta_J_eq. So we need the columns of Z.
    # Actually, we need the whole Z_mn matrix for the diagnosis equation Z_mn * j_n = delta_V_m.
    # So we compute Z_mn by solving Y.T * X = I_acc
    I_acc = np.zeros((circuit.num_free_nodes, len(acc_indices)))
    for i, idx in enumerate(acc_indices):
        I_acc[idx, i] = 1.0
    
    # Y is symmetric for R networks, so Y.T = Y.
    # We want rows of Z, which are columns of Z.T = (Y^-1).T = (Y.T)^-1 = Y^-1.
    # So we solve Y * Z_rows = I_acc to get columns that are rows of Z.
    Z_rows = spla.spsolve(Y_nom, I_acc)
    if hasattr(Z_rows, "toarray"):
        Z_rows = Z_rows.toarray()
    elif hasattr(Z_rows, "todense"):
        Z_rows = np.asarray(Z_rows.todense())
        
    Z_mn = Z_rows.T # Now it is (len(acc_indices), num_free_nodes)
    
    delta_v_ms = []
    for J in excitations:
        v_nom = solve_response(Y_nom, J)
        v_faulty = solve_response(Y_f, J)
        delta_v = v_faulty - v_nom
        delta_v_m = delta_v[acc_indices]
        delta_v_ms.append(delta_v_m)
        
    return delta_v_ms, Z_mn
