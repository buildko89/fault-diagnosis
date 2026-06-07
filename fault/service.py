"""Framework-agnostic service layer for the fault-diagnosis tool.

This module exposes the three evaluation flows (testability / diagnose /
evaluate) as plain functions that take already-built objects and return JSON-ish
``dict`` results. It carries no I/O policy of its own: it never prints, never
calls ``sys.exit``, and lets exceptions propagate so the caller (CLI or GUI)
decides how to report them.

Both ``fault/cli.py`` and the Streamlit GUI (``fault_gui/``) are thin wrappers
over the functions here, so the two front-ends share identical behavior.
"""
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .schema import (
    CircuitConfig,
    load_circuit_yaml,
    load_circuit_yaml_text,
)
from .circuit import Circuit
from .testability import check_k_node_testability
from .simulate import calculate_measurements
from .diagnose import diagnose_node_faults, reconstruct_branch_faults
from .evaluate import run_evaluation


# ---------------------------------------------------------------------------
# Input parsing (shared with the CLI)
# ---------------------------------------------------------------------------
def parse_faults(fault_str: Optional[str]) -> Dict[str, float]:
    """Parse a fault string ``"R3=0.1,R4=0.5"`` into ``{"R3": 0.1, "R4": 0.5}``.

    Raises ``ValueError`` on malformed input (the caller reports it).
    """
    if not fault_str:
        return {}
    faults: Dict[str, float] = {}
    for part in fault_str.split(','):
        if '=' not in part:
            raise ValueError(f"Invalid fault format: '{part}'. Use Name=Value.")
        name, val = part.split('=')
        faults[name.strip()] = float(val)
    return faults


def parse_freqs(freq_str: Optional[str]) -> Optional[List[float]]:
    """Parse a comma-separated list of frequencies [Hz]. Returns None if empty."""
    if not freq_str:
        return None
    return [float(f) for f in freq_str.split(',')]


def resolve_frequencies(config: CircuitConfig,
                        freq_arg: Optional[str]) -> Optional[List[float]]:
    """``--freq`` (or a GUI field) overrides; otherwise fall back to the
    circuit's configured frequencies."""
    return parse_freqs(freq_arg) or (config.frequencies or None)


# ---------------------------------------------------------------------------
# Circuit loading & summary
# ---------------------------------------------------------------------------
def load_circuit(source: str, *, is_text: bool = False) -> Circuit:
    """Load and build a Circuit from a YAML file path or inline YAML text.

    ``is_text=True`` forces ``source`` to be treated as YAML content. Otherwise,
    if ``source`` names an existing file it is read from disk, else it is parsed
    as YAML text (convenient for GUI uploads / pasted content).
    """
    if is_text:
        config = load_circuit_yaml_text(source)
    elif isinstance(source, str) and os.path.exists(source):
        config = load_circuit_yaml(source)
    else:
        config = load_circuit_yaml_text(source)
    return Circuit(config)


def circuit_summary(circuit: Circuit) -> Dict[str, Any]:
    """Return a display-friendly summary of the circuit's topology."""
    cfg = circuit.config
    return {
        "name": cfg.name,
        "reference": cfg.reference,
        "nodes": list(cfg.nodes),
        "accessible": list(cfg.accessible),
        "inaccessible": circuit.get_inaccessible_nodes(),
        "frequencies": list(cfg.frequencies) if cfg.frequencies else None,
        "num_nodes": cfg.nodes and len(cfg.nodes),
        "num_elements": len(cfg.elements),
        "elements": [
            {"name": el.name, "type": el.type, "n1": el.n1, "n2": el.n2, "value": el.value}
            for el in cfg.elements
        ],
    }


def _unit_excitations(circuit: Circuit) -> List[np.ndarray]:
    """Build one unit-current injection per accessible (non-reference) node.

    Identical to the excitation set used by the CLI and run_evaluation, so the
    three flows stay consistent.
    """
    excitations = []
    for idx in circuit.get_accessible_indices():
        J = np.zeros(circuit.num_free_nodes)
        J[idx] = 1.0
        excitations.append(J)
    return excitations


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------
def run_testability(circuit: Circuit, k: int) -> Dict[str, Any]:
    """k-node testability check."""
    testable, conns = check_k_node_testability(circuit, k)
    return {
        "name": circuit.config.name,
        "k": k,
        "testable": testable,
        # JSON-friendly keys (node ids as plain ints).
        "connectivities": {int(node): int(c) for node, c in conns.items()},
    }


def run_diagnose(
    circuit: Circuit,
    faults: Dict[str, float],
    k: int,
    *,
    top_n: int = 5,
    method: str = "auto",
    frequencies: Optional[List[float]] = None,
    reconstruct: bool = False,
) -> Dict[str, Any]:
    """Simulate the given faults and diagnose the faulty node support.

    Returns the raw ``diagnose_node_faults`` result augmented with the
    ``frequencies`` used and, when ``reconstruct`` is True, a ``branch`` entry
    holding the per-branch admittance deviation / R-C-L classification.
    """
    excitations = _unit_excitations(circuit)
    blocks = calculate_measurements(
        circuit, excitations, faulty_elements=faults, frequencies=frequencies
    )
    result = diagnose_node_faults(
        circuit, blocks, None, k, top_n=top_n, method=method
    )
    result["frequencies"] = list(frequencies) if frequencies else None
    # Voltage deviation at accessible nodes under the first excitation, kept for
    # visualization (the GUI plots it). ndarray, complex for AC. Not used by CLI.
    result["delta_v_m"] = np.asarray(blocks[0].delta_v_ms[0])

    if reconstruct and result.get("best") and result["best"]["support"]:
        support = sorted(result["best"]["support"])
        result["branch"] = reconstruct_branch_faults(
            circuit, support, excitations, blocks, result
        )
    return result


def run_evaluate(
    circuit: Circuit,
    faults: Dict[str, float],
    k: int,
    *,
    trials: int = 100,
    tol: float = 0.0,
    noise: float = 0.0,
    seed: int = 42,
    frequencies: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Monte Carlo accuracy evaluation under tolerance + measurement noise."""
    return run_evaluation(
        circuit, faults, k,
        trials=trials, tol_percent=tol, noise_std=noise, seed=seed,
        frequencies=frequencies,
    )
