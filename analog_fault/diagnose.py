import math
import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
from sklearn.linear_model import Ridge
import itertools
from typing import List, Dict, Any
from .circuit import AnalogCircuit


def _build_candidate(
    circuit: AnalogCircuit,
    Z_mn: np.ndarray,
    delta_v_ms: List[np.ndarray],
    combo: List[int],
    rcond: float,
) -> Dict[str, Any]:
    """
    Given a candidate support (list of column indices into Z_mn), solve the
    least-squares fit over all excitations and package the diagnostic metrics.
    Shared by both the OMP and exhaustive paths to avoid duplication.
    """
    combo = list(combo)
    num_excitations = len(delta_v_ms)
    B = np.column_stack(delta_v_ms)
    total_b_norm = np.sqrt(sum(la.norm(b) ** 2 for b in delta_v_ms))

    if len(combo) > 0:
        Z_S = Z_mn[:, combo]
        X_est, residuals, rank, s = la.lstsq(Z_S, B, cond=rcond)

        total_residual = 0.0
        for p in range(num_excitations):
            proj_val = Z_S @ X_est[:, p]
            total_residual += la.norm(delta_v_ms[p] - proj_val) ** 2
        total_residual = np.sqrt(total_residual)

        if len(s) > 0:
            cond = s[0] / s[-1] if s[-1] > 1e-15 else float('inf')
            min_s = s[-1]
        else:
            cond = float('inf')
            min_s = 0.0
    else:
        X_est = np.zeros((0, num_excitations))
        total_residual = total_b_norm
        cond = float('inf')
        min_s = 0.0
        rank = 0

    relative_residual = total_residual / total_b_norm if total_b_norm > 1e-12 else total_residual
    support_nodes = [circuit.idx_to_node[circuit.free_indices[idx]] for idx in combo]

    return {
        "support": support_nodes,
        "residual_norm": float(total_residual),
        "relative_residual": float(relative_residual),
        "rank": int(rank),
        "condition_number": float(cond),
        "min_singular_value": float(min_s),
        "coefficients": X_est.tolist(),
    }


def _somp_support(
    Z_mn: np.ndarray,
    delta_v_ms: List[np.ndarray],
    max_k: int,
    rcond: float,
    residual_tol: float,
) -> List[int]:
    """
    Simultaneous Orthogonal Matching Pursuit (S-OMP) for the multiple-measurement
    (multiple-excitation) problem Z_mn @ J = [ΔV_1 ... ΔV_p].

    Unlike applying scikit-learn's single-vector OMP to ``sum(|ΔV|)`` (which
    destroys the sign/linearity of the system), S-OMP enforces a *joint* support
    across all excitation columns by scoring atoms on the summed, column-norm
    normalized correlation with the residual. It also stops early once the
    relative residual drops below ``residual_tol``, so it never selects more
    faulty nodes than the data actually supports (no over-selection).
    """
    B = np.column_stack(delta_v_ms)
    b_norm = la.norm(B)
    if b_norm <= 1e-12:
        # No measurable deviation -> no fault.
        return []

    col_norms = np.linalg.norm(Z_mn, axis=0)
    safe_col_norms = np.where(col_norms > 1e-15, col_norms, np.inf)

    residual = B.copy()
    support: List[int] = []

    for _ in range(max_k):
        # Joint correlation score across all excitation columns.
        corr = np.abs(Z_mn.T @ residual) / safe_col_norms[:, None]
        score = corr.sum(axis=1)
        if support:
            score[support] = -np.inf
        j = int(np.argmax(score))
        if not np.isfinite(score[j]) or score[j] <= 0.0:
            break
        support.append(j)

        Z_S = Z_mn[:, support]
        X, _res, _rank, _s = la.lstsq(Z_S, B, cond=rcond)
        residual = B - Z_S @ X

        if la.norm(residual) / b_norm <= residual_tol:
            break

    return sorted(support)

def diagnose_node_faults(
    circuit: AnalogCircuit,
    Z_mn: np.ndarray,
    delta_v_ms: List[np.ndarray],
    max_faults: int,
    top_n: int = 5,
    ambiguity_ratio: float = 1.2,
    rcond: float = 1e-10,
    method: str = 'auto',
    omp_residual_tol: float = 1e-6,
    exhaustive_max_combos: int = 100_000
) -> Dict[str, Any]:
    """
    Diagnoses node faults by solving Z_mn * j_n = delta_v_m.
    Supports multiple excitations.

    method:
        'auto'       - (default) use the exact 'exhaustive' search when the
                       combination count is small enough (<= exhaustive_max_combos),
                       otherwise fall back to the fast approximate 'omp'.
        'exhaustive' - try every node combination up to max_faults; exact but
                       O(nCk). Reliable even on highly coherent/symmetric circuits.
        'omp'        - Simultaneous Orthogonal Matching Pursuit; a fast greedy
                       approximation. NOTE: on coherent or symmetric topologies
                       (e.g. a bridge) greedy/L1 sparse methods can mislocate
                       faults - especially internal (inaccessible) nodes - so
                       prefer 'exhaustive'/'auto' for small circuits.

    For the OMP path, ``omp_residual_tol`` is the relative-residual threshold at
    which the greedy search stops adding faulty nodes. This prevents reporting
    more faults than the data supports when the true fault count < max_faults.
    """
    m, n = Z_mn.shape

    # A node fault can only be resolved if there are enough independent
    # measurements; selecting more than m faulty nodes is not identifiable.
    max_k = min(max_faults, m, n)
    if max_faults > m:
        warnings.warn(
            f"max_faults ({max_faults}) exceeds the number of measurements "
            f"({m}); the search is limited to {max_k} simultaneous faults.",
            stacklevel=2,
        )

    # Resolve 'auto' to the exact search when affordable, else the fast greedy.
    resolved_method = method
    if method == 'auto':
        num_combos = sum(math.comb(n, r) for r in range(1, max_k + 1))
        resolved_method = 'exhaustive' if num_combos <= exhaustive_max_combos else 'omp'

    candidates = []

    if resolved_method == 'omp':
        support = _somp_support(Z_mn, delta_v_ms, max_k, rcond, omp_residual_tol)
        candidates.append(_build_candidate(circuit, Z_mn, delta_v_ms, support, rcond))

    elif resolved_method == 'exhaustive':
        for r in range(1, max_k + 1):
            for combo in itertools.combinations(range(n), r):
                candidates.append(
                    _build_candidate(circuit, Z_mn, delta_v_ms, combo, rcond)
                )
    else:
        raise ValueError(f"Unknown method {method}")
            
    candidates.sort(key=lambda x: x["residual_norm"])
    
    if not candidates:
        return {"status": "no_candidates", "best": None, "candidates": []}
    
    best = candidates[0]
    top_results = candidates[:top_n]
    
    status = "unique"
    ambiguous_candidates = []
    
    if len(candidates) > 1:
        noise_floor = 1e-12
        is_ambiguous = (candidates[1]["residual_norm"] < best["residual_norm"] * ambiguity_ratio) or \
                       (candidates[1]["residual_norm"] < noise_floor and best["residual_norm"] < noise_floor)
        
        if is_ambiguous:
            status = "ambiguous"
            ambiguous_candidates = [c for c in candidates[1:] 
                                    if (c["residual_norm"] < best["residual_norm"] * ambiguity_ratio) or
                                       (c["residual_norm"] < noise_floor and best["residual_norm"] < noise_floor)]
            
    if status == "unique" and best["support"] and best["condition_number"] > 1.0 / rcond:
        status = "ill_conditioned"
        
    return {
        "status": status,
        "best": best,
        "candidates": top_results,
        "ambiguous_candidates": ambiguous_candidates
    }

def reconstruct_branch_faults(
    circuit: AnalogCircuit,
    faulty_nodes: List[int],
    excitations: List[np.ndarray],
    delta_v_ms: List[np.ndarray],
    diagnose_result: Dict[str, Any],
    method: str = 'lstsq',
    alpha: float = 1.0
) -> Dict[str, Any]:
    """
    Reconstructs branch admittance deviations (Delta g) for the given faulty nodes.
    Supports method='lstsq' or method='ridge' (for L2 regularization).
    """
    if not diagnose_result.get("best"):
        return {}
        
    best = diagnose_result["best"]
    support_nodes = best["support"]
    
    combo_indices = [circuit.global_to_reduced[circuit.node_to_idx[node]] for node in support_nodes]
    
    A_nom, Yb_nom, Y_nom = circuit.build_matrices(faulty_elements=None)
    
    candidate_branch_indices = []
    for idx, el in enumerate(circuit.config.elements):
        if el.n1 in faulty_nodes or el.n2 in faulty_nodes:
            candidate_branch_indices.append(idx)
            
    num_candidates = len(candidate_branch_indices)
    if num_candidates == 0:
        return {}
        
    eq_rows = []
    eq_b = []
    
    coeffs_matrix = np.array(best["coefficients"]) # shape: (r, num_excitations)
    
    for p in range(len(excitations)):
        J = excitations[p]
        delta_v_m = delta_v_ms[p]
        
        j_n_est = np.zeros(circuit.num_free_nodes)
        for i, row_idx in enumerate(combo_indices):
            j_n_est[row_idx] = coeffs_matrix[i, p]
            
        v_nom = spla.spsolve(Y_nom, J)
        delta_v_est = spla.spsolve(Y_nom, j_n_est)
        
        v_f_est = v_nom + delta_v_est
        v_bf_est = A_nom.T @ v_f_est
        
        for node_i in faulty_nodes:
            if node_i == circuit.reference_node:
                continue
            row_idx = circuit.global_to_reduced[circuit.node_to_idx[node_i]]
            rhs = j_n_est[row_idx]
            
            coeffs = []
            for e_idx in candidate_branch_indices:
                coeff = - A_nom[row_idx, e_idx] * v_bf_est[e_idx]
                coeffs.append(coeff)
                
            eq_rows.append(coeffs)
            eq_b.append(rhs)
            
    eq_rows = np.array(eq_rows)
    eq_b = np.array(eq_b)
    
    if method == 'ridge':
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(eq_rows, eq_b)
        delta_g_est = ridge.coef_
    else:
        delta_g_est, residuals, rank, s = la.lstsq(eq_rows, eq_b, cond=1e-10)
    
    results = {}
    for i, e_idx in enumerate(candidate_branch_indices):
        el = circuit.config.elements[e_idx]
        results[el.name] = {
            'branch_idx': e_idx,
            'nominal_g': el.value,
            'delta_g': delta_g_est[i],
            'estimated_g': el.value + delta_g_est[i]
        }
        
    return results
