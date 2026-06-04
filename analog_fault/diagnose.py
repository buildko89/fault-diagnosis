import math
import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
from sklearn.linear_model import Ridge
import itertools
from typing import List, Dict, Any, Tuple
from .circuit import AnalogCircuit


def _as_blocks(Z_mn, delta_v_ms) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Normalize diagnosis inputs into a list of (Z, B) blocks, where B is the
    column-stacked measurement matrix (m x p). Accepts either:
      - a list of MeasurementBlock-like objects (attributes ``Z_mn`` and
        ``delta_v_ms``) for the multi-frequency case, or
      - the legacy single-block form (Z_mn ndarray, delta_v_ms list).
    """
    if isinstance(Z_mn, (list, tuple)):
        return [(np.asarray(blk.Z_mn), np.column_stack(blk.delta_v_ms)) for blk in Z_mn]
    return [(np.asarray(Z_mn), np.column_stack(delta_v_ms))]


def _build_candidate(
    circuit: AnalogCircuit,
    blocks: List[Tuple[np.ndarray, np.ndarray]],
    combo: List[int],
    rcond: float,
) -> Dict[str, Any]:
    """
    Given a candidate support (list of column indices into Z), solve the
    least-squares fit and package the diagnostic metrics. Coefficients are fit
    independently per block (per frequency), while the support is shared; the
    residual is the root-sum-square across all blocks and excitations.
    Shared by both the OMP and exhaustive paths to avoid duplication.
    """
    combo = list(combo)
    num_excitations = blocks[0][1].shape[1]
    total_b_norm = np.sqrt(sum(la.norm(B) ** 2 for _, B in blocks))

    if len(combo) > 0:
        total_residual_sq = 0.0
        worst_cond = 0.0
        min_s_overall = np.inf
        min_rank = None
        coeffs_block0 = None

        for bi, (Z, B) in enumerate(blocks):
            Z_S = Z[:, combo]
            X_est, _residuals, rank, s = la.lstsq(Z_S, B, cond=rcond)
            total_residual_sq += la.norm(B - Z_S @ X_est) ** 2

            if len(s) > 0:
                cond = s[0] / s[-1] if s[-1] > 1e-15 else float('inf')
                worst_cond = max(worst_cond, cond)
                min_s_overall = min(min_s_overall, s[-1])
            else:
                worst_cond = float('inf')
                min_s_overall = 0.0

            min_rank = rank if min_rank is None else min(min_rank, rank)
            if bi == 0:
                coeffs_block0 = X_est

        total_residual = np.sqrt(total_residual_sq)
        cond = worst_cond
        min_s = min_s_overall if np.isfinite(min_s_overall) else 0.0
        rank = int(min_rank)
        X_out = coeffs_block0
    else:
        total_residual = total_b_norm
        cond = float('inf')
        min_s = 0.0
        rank = 0
        X_out = np.zeros((0, num_excitations))

    relative_residual = total_residual / total_b_norm if total_b_norm > 1e-12 else total_residual
    support_nodes = [circuit.idx_to_node[circuit.free_indices[idx]] for idx in combo]

    return {
        "support": support_nodes,
        "residual_norm": float(total_residual),
        "relative_residual": float(relative_residual),
        "rank": int(rank),
        "condition_number": float(cond),
        "min_singular_value": float(min_s),
        # Per-block-0 coefficients (= the single block in the DC case), shape (r, p).
        "coefficients": np.asarray(X_out).tolist(),
    }


def _somp_support(
    blocks: List[Tuple[np.ndarray, np.ndarray]],
    max_k: int,
    rcond: float,
    residual_tol: float,
) -> List[int]:
    """
    Simultaneous Orthogonal Matching Pursuit (S-OMP) for the multi-block,
    multi-excitation problem Z^(f) @ J^(f) = B^(f) (one block per frequency).

    The support is selected *jointly* across all blocks and excitation columns by
    scoring atoms on the summed, column-norm-normalized correlation with the
    residual (using the conjugate transpose, so complex/AC data is handled
    correctly). Coefficients are then re-fit independently per block. The search
    stops once the relative residual drops below ``residual_tol``, so it never
    selects more faulty nodes than the data supports (no over-selection).
    """
    b_norm_total = np.sqrt(sum(la.norm(B) ** 2 for _, B in blocks))
    if b_norm_total <= 1e-12:
        # No measurable deviation -> no fault.
        return []

    n = blocks[0][0].shape[1]
    safe_col_norms = []
    residuals = []
    for Z, B in blocks:
        col_norms = np.linalg.norm(Z, axis=0)
        safe_col_norms.append(np.where(col_norms > 1e-15, col_norms, np.inf))
        residuals.append(B.copy())

    support: List[int] = []

    for _ in range(max_k):
        # Joint correlation score across all blocks and excitation columns.
        score = np.zeros(n)
        for (Z, _B), sn, residual in zip(blocks, safe_col_norms, residuals):
            corr = np.abs(Z.conj().T @ residual) / sn[:, None]
            score += corr.sum(axis=1)
        if support:
            score[support] = -np.inf
        j = int(np.argmax(score))
        if not np.isfinite(score[j]) or score[j] <= 0.0:
            break
        support.append(j)

        # Independent per-block (per-frequency) refit and residual update.
        res_total_sq = 0.0
        for bi, (Z, B) in enumerate(blocks):
            Z_S = Z[:, support]
            X, _res, _rank, _s = la.lstsq(Z_S, B, cond=rcond)
            residuals[bi] = B - Z_S @ X
            res_total_sq += la.norm(residuals[bi]) ** 2

        if np.sqrt(res_total_sq) / b_norm_total <= residual_tol:
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

    Inputs may be either the legacy single-block form (``Z_mn`` ndarray +
    ``delta_v_ms`` list) or a list of MeasurementBlock objects passed as ``Z_mn``
    (for multi-frequency / AC diagnosis); in the latter case ``delta_v_ms`` is
    ignored. Multiple frequencies add independent constraints on the support and
    improve identifiability on coherent/symmetric circuits.
    """
    blocks = _as_blocks(Z_mn, delta_v_ms)
    m, n = blocks[0][0].shape

    # A node fault can only be resolved if there are enough independent
    # measurements per frequency; selecting more than m faulty nodes is not
    # identifiable.
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
        support = _somp_support(blocks, max_k, rcond, omp_residual_tol)
        candidates.append(_build_candidate(circuit, blocks, support, rcond))

    elif resolved_method == 'exhaustive':
        for r in range(1, max_k + 1):
            for combo in itertools.combinations(range(n), r):
                candidates.append(
                    _build_candidate(circuit, blocks, combo, rcond)
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

def _injection_blocks(circuit, measurements, best, combo_indices, rcond):
    """
    Returns a list of (omega, coeffs_matrix) where coeffs_matrix (r x p) holds the
    estimated fault-injection currents at the support nodes for each excitation.

    For multi-frequency input (a list of MeasurementBlock), the injections are
    re-fit per frequency from (Z, ΔV). For the legacy DC form (delta_v_ms list),
    the stored diagnosis coefficients are reused (omega=None).
    """
    is_blocks = (
        isinstance(measurements, (list, tuple)) and len(measurements) > 0
        and hasattr(measurements[0], "Z_mn")
    )
    if is_blocks:
        blocks = []
        for blk in measurements:
            Z_S = np.asarray(blk.Z_mn)[:, combo_indices]
            B = np.column_stack(blk.delta_v_ms)
            X, _r, _rk, _s = la.lstsq(Z_S, B, cond=rcond)
            blocks.append((blk.omega, X))
        return blocks
    return [(None, np.asarray(best["coefficients"]))]


def _solve_deviation(eq_rows, eq_b, method, alpha):
    """Solve eq_rows @ x = eq_b for the branch admittance deviations."""
    if method == 'ridge':
        if np.iscomplexobj(eq_rows) or np.iscomplexobj(eq_b):
            warnings.warn(
                "method='ridge' is not supported for complex (AC) reconstruction; "
                "falling back to least squares.",
                stacklevel=3,
            )
        else:
            ridge = Ridge(alpha=alpha, fit_intercept=False)
            ridge.fit(eq_rows, eq_b)
            return ridge.coef_
    delta, _residuals, _rank, _s = la.lstsq(eq_rows, eq_b, cond=1e-10)
    return delta


def _classify_branch_deviation(delta_y_by_freq: Dict[float, complex]) -> Dict[str, Any]:
    """
    Given a branch's admittance deviation across frequencies, fit
        Δy(ω) = ΔG·1 + ΔC·(jω) + Δ(1/L)·(1/(jω))
    and classify the dominant fault nature (R / C / L) from the basis that best
    explains the frequency dependence.
    """
    omegas = np.array(list(delta_y_by_freq.keys()), dtype=float)
    y = np.array(list(delta_y_by_freq.values()), dtype=complex)

    # Complex design matrix [1, jω, 1/(jω)] solved for real unknowns via real/imag stacking.
    M = np.column_stack([np.ones_like(omegas, dtype=complex),
                         1j * omegas,
                         1.0 / (1j * omegas)])
    M_real = np.vstack([M.real, M.imag])
    y_real = np.concatenate([y.real, y.imag])
    x, _res, _rank, _s = la.lstsq(M_real, y_real, cond=1e-10)
    delta_G, delta_C, delta_invL = (float(x[0]), float(x[1]), float(x[2]))

    # Compare each basis' contribution magnitude across the measured frequencies.
    contrib = {
        'R': abs(delta_G) * np.linalg.norm(np.ones_like(omegas)),
        'C': abs(delta_C) * np.linalg.norm(omegas),
        'L': abs(delta_invL) * np.linalg.norm(1.0 / omegas),
    }
    classification = max(contrib, key=contrib.get)
    delta_value = {'R': delta_G, 'C': delta_C, 'L': delta_invL}[classification]

    return {
        'classification': classification,
        'delta_G': delta_G,
        'delta_C': delta_C,
        'delta_invL': delta_invL,
        'delta_value': delta_value,
    }


def reconstruct_branch_faults(
    circuit: AnalogCircuit,
    faulty_nodes: List[int],
    excitations: List[np.ndarray],
    measurements,
    diagnose_result: Dict[str, Any],
    method: str = 'lstsq',
    alpha: float = 1.0,
    rcond: float = 1e-10,
) -> Dict[str, Any]:
    """
    Reconstructs branch admittance deviations for the diagnosed faulty nodes.

    ``measurements`` may be the legacy ``delta_v_ms`` list (DC) or a list of
    MeasurementBlock objects (multi-frequency / AC). For DC, results carry the
    real admittance deviation as 'delta_g' (backward compatible). For AC, results
    carry the per-frequency complex deviation 'delta_y_by_freq' and, when >= 2
    frequencies are available, a R/C/L classification of the fault.

    method='lstsq' (default) or 'ridge' (L2 regularization, real/DC only).
    """
    if not diagnose_result.get("best"):
        return {}

    best = diagnose_result["best"]
    support_nodes = best["support"]
    combo_indices = [circuit.global_to_reduced[circuit.node_to_idx[node]] for node in support_nodes]

    candidate_branch_indices = [
        idx for idx, el in enumerate(circuit.config.elements)
        if el.n1 in faulty_nodes or el.n2 in faulty_nodes
    ]
    if len(candidate_branch_indices) == 0:
        return {}

    inj_blocks = _injection_blocks(circuit, measurements, best, combo_indices, rcond)

    # Estimate the branch admittance deviation at each frequency.
    delta_y_by_freq: Dict[Any, np.ndarray] = {}
    for omega, coeffs_matrix in inj_blocks:
        coeffs_matrix = np.asarray(coeffs_matrix)
        A_nom, _Yb_nom, Y_nom = circuit.build_matrices(faulty_elements=None, omega=omega)

        eq_rows, eq_b = [], []
        for p in range(len(excitations)):
            J = excitations[p]
            j_n_est = np.zeros(circuit.num_free_nodes, dtype=coeffs_matrix.dtype)
            for i, row_idx in enumerate(combo_indices):
                j_n_est[row_idx] = coeffs_matrix[i, p]

            v_nom = spla.spsolve(Y_nom, J)
            delta_v_est = spla.spsolve(Y_nom, j_n_est)
            v_bf_est = A_nom.T @ (v_nom + delta_v_est)

            for node_i in faulty_nodes:
                if node_i == circuit.reference_node:
                    continue
                row_idx = circuit.global_to_reduced[circuit.node_to_idx[node_i]]
                eq_b.append(j_n_est[row_idx])
                eq_rows.append([-A_nom[row_idx, e_idx] * v_bf_est[e_idx]
                                for e_idx in candidate_branch_indices])

        delta_y_by_freq[omega] = _solve_deviation(np.array(eq_rows), np.array(eq_b), method, alpha)

    is_dc = len(inj_blocks) == 1 and inj_blocks[0][0] is None

    results = {}
    for i, e_idx in enumerate(candidate_branch_indices):
        el = circuit.config.elements[e_idx]
        if is_dc:
            delta_g = float(np.real(delta_y_by_freq[None][i]))
            results[el.name] = {
                'branch_idx': e_idx,
                'nominal_g': el.value,
                'delta_g': delta_g,
                'estimated_g': el.value + delta_g,
            }
        else:
            dy = {float(omega): complex(delta_y_by_freq[omega][i]) for omega in delta_y_by_freq}
            entry = {
                'branch_idx': e_idx,
                'type': el.type,
                'nominal_value': el.value,
                'delta_y_by_freq': dy,
            }
            if len(dy) >= 2:
                entry.update(_classify_branch_deviation(dy))
            results[el.name] = entry

    return results
