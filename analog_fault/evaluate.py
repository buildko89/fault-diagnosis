import warnings
import numpy as np
from typing import Dict, Any
from .circuit import AnalogCircuit
from .simulate import calculate_delta_v
from .diagnose import diagnose_node_faults

def run_evaluation(
    circuit: AnalogCircuit,
    faulty_elements: Dict[str, float],
    max_faults: int,
    trials: int = 100,
    tol_percent: float = 0.0,
    noise_std: float = 0.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Runs Monte Carlo evaluation with tolerances and noise.
    """
    rng = np.random.default_rng(seed)
    
    results = []
    
    # Pre-determine excitations
    acc_indices = circuit.get_accessible_indices()
    excitations = []
    for idx in acc_indices:
        J = np.zeros(circuit.num_free_nodes)
        J[idx] = 1.0
        excitations.append(J)
        
    correct_nodes_set = set()
    for el_name in faulty_elements:
        for el in circuit.config.elements:
            if el.name == el_name:
                if el.n1 != circuit.reference_node:
                    correct_nodes_set.add(el.n1)
                if el.n2 != circuit.reference_node:
                    correct_nodes_set.add(el.n2)
    
    # 2.5 Improvement: Warn if actual faulty nodes exceed search capacity
    if len(correct_nodes_set) > max_faults:
        warnings.warn(
            f"The number of actual faulty nodes ({len(correct_nodes_set)}) "
            f"exceeds max_faults ({max_faults}); accuracy will likely be 0%.",
            stacklevel=2,
        )

    top1_hits = 0
    top3_hits = 0
    ambiguous_count = 0
    ill_conditioned_count = 0
    
    for _ in range(trials):
        delta_v_ms, Z_mn = calculate_delta_v(
            circuit, excitations, faulty_elements=faulty_elements, rng=rng, tol_percent=tol_percent
        )
        
        if noise_std > 0:
            for i in range(len(delta_v_ms)):
                delta_v_ms[i] += rng.normal(0, noise_std, delta_v_ms[i].shape)
        
        diagnosis = diagnose_node_faults(circuit, Z_mn, delta_v_ms, max_faults)
        
        best_support = set(diagnosis["best"]["support"]) if diagnosis["best"] else set()
        
        if best_support == correct_nodes_set:
            top1_hits += 1
            
        for cand in diagnosis["candidates"][:3]:
            if set(cand["support"]) == correct_nodes_set:
                top3_hits += 1
                break
        
        if diagnosis["status"] == "ambiguous":
            ambiguous_count += 1
        if diagnosis["status"] == "ill_conditioned":
            ill_conditioned_count += 1
            
        results.append(diagnosis)
        
    return {
        "trials": trials,
        "tol_percent": tol_percent,
        "noise_std": noise_std,
        "top1_accuracy": top1_hits / trials,
        "top3_accuracy": top3_hits / trials,
        "ambiguous_rate": ambiguous_count / trials,
        "ill_conditioned_rate": ill_conditioned_count / trials,
        "correct_nodes": list(correct_nodes_set)
    }
