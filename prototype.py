import numpy as np
from fault.schema import CircuitConfig, Element
from fault.circuit import Circuit
from fault.testability import check_k_node_testability
from fault.simulate import calculate_delta_v
from fault.diagnose import diagnose_node_faults, reconstruct_branch_faults
import time
from fault.reporter import generate_markdown_report

if __name__ == "__main__":
    print("==============================================================")
    print("Fault Diagnosis Prototype Demonstration")
    print("Based on Huang-Lin-Liu (1983) and Togawa-Matsumoto (1984)")
    print("==============================================================")
    
    # --------------------------------------------------------
    # DEMO 1: Ladder Network (Reconstructibility Limits)
    # --------------------------------------------------------
    print("\n=== DEMO 1: Ladder Network (Reconstructibility Limits) ===")
    
    config1 = CircuitConfig(
        name="ladder_demo",
        reference=0,
        nodes=[0, 1, 2, 3, 4, 5],
        accessible=[1, 2, 3],
        elements=[
            Element("R1", "R", 1, 0, 1.0),
            Element("R2", "R", 1, 4, 2.0),
            Element("R3", "R", 4, 0, 1.5),
            Element("R4", "R", 4, 2, 1.0),
            Element("R5", "R", 2, 5, 2.0),
            Element("R6", "R", 5, 0, 1.5),
            Element("R7", "R", 5, 3, 1.0),
            Element("R8", "R", 3, 0, 1.0),
        ]
    )
    circuit1 = Circuit(config1)
    
    print("\n--- 1. Topological Analysis ---")
    print(f"Accessible nodes (excluding ground): {config1.accessible}")
    print(f"Inaccessible nodes: {circuit1.get_inaccessible_nodes()}")
    
    for k in [1, 2]:
        testable, conns = check_k_node_testability(circuit1, k)
        print(f"Is {k}-node fault testable? {testable}")
        print(f"  Node connectivities: {conns}")
        
    print("\n--- 2. Fault Simulation ---")
    faulty_branches1 = {"R3": 0.1}
    print("Simulating fault: R3 drops from 1.5 to 0.1 (conductance)")
    tolerance_pct = 2.0
    print(f"Applying a random parameter tolerance of {tolerance_pct}% to all healthy branches.")
    
    excitations1 = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    ]
    
    # Fixed seed Generator instead of np.random.seed()
    rng1 = np.random.default_rng(42)
    delta_v_ms1, Z_mn1 = calculate_delta_v(
        circuit1, excitations1, faulty_elements=faulty_branches1, rng=rng1, tol_percent=tolerance_pct
    )
    
    print("\n--- 3. Fault Diagnosis (Locating Faulty Nodes) ---")
    k_diag1 = 1
    print(f"Running k-node fault diagnosis with k = {k_diag1}:")
    
    result1 = diagnose_node_faults(circuit1, Z_mn1, delta_v_ms1, k_diag1)
    best1 = result1["best"]
    consensus_faulty_nodes1 = sorted(best1["support"])
    
    print(f"  Diagnosed faulty nodes = {consensus_faulty_nodes1}, projection error = {best1['residual_norm']:.6f}")
    print(f"Consensus diagnosed faulty nodes: {consensus_faulty_nodes1}")
    
    print("\n--- 4. Branch Admittance Reconstruction ---")
    branch_results1 = reconstruct_branch_faults(
        circuit1, consensus_faulty_nodes1, excitations1, delta_v_ms1, result1
    )
    
    print("Reconstruction Results:")
    print(f"{'Branch':<8} | {'Nominal G':<10} | {'Estimated G':<12} | {'Estimated Delta G':<18} | {'Actual Delta G':<15}")
    print("-" * 75)
    for name, res in branch_results1.items():
        actual_delta = 0.0
        if name in faulty_branches1:
            actual_delta = faulty_branches1[name] - res['nominal_g']
        print(f"{name:<8} | {res['nominal_g']:<10.4f} | {res['estimated_g']:<12.4f} | {res['delta_g']:<18.4f} | {actual_delta:<15.4f}")
    print("\nNote: Reconstruction fails to match actual values due to rank deficiency (physical reconstructibility limits).")
    
    # --------------------------------------------------------
    # DEMO 2: Bridge Network (Complete Reconstruction)
    # --------------------------------------------------------
    print("\n=== DEMO 2: Bridge Network (Complete Admittance Reconstruction) ===")
    
    config2 = CircuitConfig(
        name="bridge_demo",
        reference=0,
        nodes=[0, 1, 2, 3, 4],
        accessible=[1, 2, 3],
        elements=[
            Element("R1", "R", 1, 0, 1.0),
            Element("R2", "R", 2, 0, 1.0),
            Element("R3", "R", 3, 0, 1.0),
            Element("R4", "R", 1, 4, 1.0),
            Element("R5", "R", 2, 4, 1.0),
            Element("R6", "R", 3, 4, 1.0),
        ]
    )
    circuit2 = Circuit(config2)
    
    print("\n--- 1. Topological Analysis ---")
    print(f"Accessible nodes (excluding ground): {config2.accessible}")
    print(f"Inaccessible nodes: {circuit2.get_inaccessible_nodes()}")
    
    for k in [1, 2]:
        testable, conns = check_k_node_testability(circuit2, k)
        print(f"Is {k}-node fault testable? {testable}")
        print(f"  Node connectivities: {conns}")
        
    print("\n--- 2. Fault Simulation ---")
    faulty_branches2 = {"R4": 0.1}
    print("Simulating fault: R4 drops from 1.0 to 0.1 (conductance)")
    print("Applying 0% parameter tolerance to verify exact theoretical reconstruction.")
    
    excitations2 = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
    ]
    
    rng2 = np.random.default_rng(42)
    delta_v_ms2, Z_mn2 = calculate_delta_v(
        circuit2, excitations2, faulty_elements=faulty_branches2, rng=rng2, tol_percent=0.0
    )
    
    print("\n--- 3. Fault Diagnosis (Locating Faulty Nodes) ---")
    k_diag2 = 2
    print(f"Running k-node fault diagnosis with k = {k_diag2}:")
    
    t0 = time.perf_counter()
    result2_omp = diagnose_node_faults(circuit2, Z_mn2, delta_v_ms2, k_diag2, method='omp')
    t1 = time.perf_counter()
    time_omp = t1 - t0
    
    t0 = time.perf_counter()
    result2_leg = diagnose_node_faults(circuit2, Z_mn2, delta_v_ms2, k_diag2, method='exhaustive')
    t1 = time.perf_counter()
    time_leg = t1 - t0
    
    result2 = result2_leg
    best2 = result2["best"]
    consensus_faulty_nodes2 = sorted(best2["support"])
    
    print(f"  Diagnosed faulty nodes (Exhaustive) = {consensus_faulty_nodes2}, projection error = {best2['residual_norm']:.6e}")
    print(f"Consensus diagnosed faulty nodes: {consensus_faulty_nodes2}")
    
    print("\n--- 4. Branch Admittance Reconstruction ---")
    branch_results2 = reconstruct_branch_faults(
        circuit2, consensus_faulty_nodes2, excitations2, delta_v_ms2, result2
    )
    
    print("Reconstruction Results:")
    print(f"{'Branch':<8} | {'Nominal G':<10} | {'Estimated G':<12} | {'Estimated Delta G':<18} | {'Actual Delta G':<15}")
    print("-" * 75)
    for name, res in branch_results2.items():
        actual_delta = 0.0
        if name in faulty_branches2:
            actual_delta = faulty_branches2[name] - res['nominal_g']
        print(f"{name:<8} | {res['nominal_g']:<10.4f} | {res['estimated_g']:<12.4f} | {res['delta_g']:<18.4f} | {actual_delta:<15.4f}")
        
        
    print("\nNote: Reconstruction is mathematically exact on the Bridge topology (full-rank system).")
    
    print("\n--- 5. Generate Markdown Report ---")
    print("Generating report using OMP vs Exhaustive results...")
    
    res_omp_dict = {'result': result2_omp, 'time': time_omp}
    res_leg_dict = {'result': result2_leg, 'time': time_leg}
    
    md_content = generate_markdown_report(circuit2, delta_v_ms2[0], res_omp_dict, res_leg_dict)
    print("Markdown report generated successfully in ./report directory.")
    print("==============================================================")
