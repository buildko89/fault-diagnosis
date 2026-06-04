import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from .circuit import AnalogCircuit


@dataclass
class MeasurementBlock:
    """
    Observations at a single angular frequency.

    omega       : angular frequency [rad/s], or None for DC.
    Z_mn        : (m, n) transfer-impedance matrix at the accessible nodes
                  (complex when omega is given).
    delta_v_ms  : list of ΔV_m vectors (one per excitation) at the accessible nodes.
    """
    omega: Optional[float]
    Z_mn: np.ndarray
    delta_v_ms: List[np.ndarray]


def solve_response(Y: sp.spmatrix, J: np.ndarray) -> np.ndarray:
    """Solves Y * V = J for V using a stable sparse solver (real or complex)."""
    return spla.spsolve(Y, J)


def _transfer_impedance(circuit: AnalogCircuit, Y_nom: sp.spmatrix,
                        acc_indices: List[int]) -> np.ndarray:
    """
    Z_mn = rows of inv(Y_nom) at the accessible nodes, computed by solving
    Y_nom * X = I_acc (Y is symmetric for reciprocal RLC networks, so its rows
    equal the corresponding columns). Returns a dense (m, n) array.
    """
    I_acc = np.zeros((circuit.num_free_nodes, len(acc_indices)), dtype=Y_nom.dtype)
    for i, idx in enumerate(acc_indices):
        I_acc[idx, i] = 1.0

    Z_rows = spla.spsolve(Y_nom, I_acc)
    if hasattr(Z_rows, "toarray"):
        Z_rows = Z_rows.toarray()
    elif hasattr(Z_rows, "todense"):
        Z_rows = np.asarray(Z_rows.todense())

    # spsolve squeezes a single-column RHS to 1-D; restore the (n, m) shape so
    # Z_mn is always 2-D, even for circuits with one free/accessible node.
    Z_rows = np.asarray(Z_rows).reshape(circuit.num_free_nodes, len(acc_indices))
    return Z_rows.T  # (m, n)


def _sample_measured_values(circuit: AnalogCircuit,
                            faulty_elements: Optional[Dict[str, float]],
                            rng: Optional[np.random.Generator],
                            tol_percent: float) -> Dict[str, float]:
    """
    Draw ONE physical realization of every element value: faults override the
    nominal value, healthy elements get a Gaussian tolerance perturbation (when
    enabled). This realization is shared across all frequencies, so a component
    keeps a single value that is then measured at every frequency.

    The draw order (iterating elements, sampling only the non-faulty ones) is
    identical to the previous in-build sampling, preserving exact RNG behavior.
    """
    faulty_elements = faulty_elements or {}
    values: Dict[str, float] = {}
    for el in circuit.config.elements:
        if el.name in faulty_elements:
            values[el.name] = faulty_elements[el.name]
        elif tol_percent > 0.0 and rng is not None:
            std_dev = (tol_percent / 100.0) * el.value
            v = el.value + rng.normal(0, std_dev)
            values[el.name] = max(v, 1e-9)  # Ensure positive
        else:
            values[el.name] = el.value
    return values


def calculate_measurements(circuit: AnalogCircuit, excitations: List[np.ndarray],
                           faulty_elements: Optional[Dict[str, float]] = None,
                           rng: Optional[np.random.Generator] = None,
                           tol_percent: float = 0.0,
                           frequencies: Optional[List[float]] = None
                           ) -> List[MeasurementBlock]:
    """
    Simulate the nominal and faulty circuit at one or more frequencies and return
    a MeasurementBlock per frequency.

    frequencies : list of measurement frequencies in Hz. None/empty -> a single
                  DC block (omega=None). Each f is converted to omega = 2*pi*f.
    """
    if frequencies:
        omegas: List[Optional[float]] = [2.0 * np.pi * f for f in frequencies]
    else:
        omegas = [None]

    # One shared realization (faults + tolerance) for all frequencies.
    measured_values = _sample_measured_values(circuit, faulty_elements, rng, tol_percent)

    acc_indices = circuit.get_accessible_indices()
    blocks: List[MeasurementBlock] = []

    for omega in omegas:
        _, _, Y_nom = circuit.build_matrices(faulty_elements=None, omega=omega)
        _, _, Y_f = circuit.build_matrices(faulty_elements=measured_values, omega=omega)

        Z_mn = _transfer_impedance(circuit, Y_nom, acc_indices)

        delta_v_ms = []
        for J in excitations:
            v_nom = solve_response(Y_nom, J)
            v_faulty = solve_response(Y_f, J)
            delta_v_m = (v_faulty - v_nom)[acc_indices]
            delta_v_ms.append(delta_v_m)

        blocks.append(MeasurementBlock(omega=omega, Z_mn=Z_mn, delta_v_ms=delta_v_ms))

    return blocks


def calculate_delta_v(circuit: AnalogCircuit, excitations: List[np.ndarray],
                      faulty_elements: Optional[Dict[str, float]] = None,
                      rng: Optional[np.random.Generator] = None,
                      tol_percent: float = 0.0) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Backward-compatible single-(DC)-block wrapper.

    Returns:
        delta_v_ms: List of deviation vectors at accessible nodes.
        Z_mn:       Transfer impedance matrix (nominal) for accessible nodes.
    """
    block = calculate_measurements(
        circuit, excitations, faulty_elements=faulty_elements,
        rng=rng, tol_percent=tol_percent, frequencies=None,
    )[0]
    return block.delta_v_ms, block.Z_mn
