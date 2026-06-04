import argparse
import sys
import json
import numpy as np
from .schema import load_circuit_yaml
from .circuit import AnalogCircuit
from .testability import check_k_node_testability
from .simulate import calculate_measurements
from .diagnose import diagnose_node_faults
from .evaluate import run_evaluation

def cmd_testability(args):
    try:
        config = load_circuit_yaml(args.config)
        circuit = AnalogCircuit(config)
        testable, conns = check_k_node_testability(circuit, args.k)
        print(f"Config: {config.name}")
        print(f"K: {args.k}")
        print(f"Testable: {testable}")
        print(f"Connectivities: {conns}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def parse_faults(fault_str):
    if not fault_str:
        return {}
    faults = {}
    try:
        for part in fault_str.split(','):
            if '=' not in part:
                raise ValueError(f"Invalid fault format: '{part}'. Use Name=Value.")
            name, val = part.split('=')
            faults[name] = float(val)
    except ValueError as ve:
        print(f"Error parsing faults: {ve}", file=sys.stderr)
        sys.exit(1)
    return faults

def parse_freqs(freq_str):
    """Parse a comma-separated list of frequencies [Hz]. Returns None if empty."""
    if not freq_str:
        return None
    try:
        return [float(f) for f in freq_str.split(',')]
    except ValueError as ve:
        print(f"Error parsing frequencies: {ve}", file=sys.stderr)
        sys.exit(1)

def resolve_frequencies(config, freq_arg):
    """CLI --freq overrides; otherwise fall back to the circuit's configured frequencies."""
    return parse_freqs(freq_arg) or (config.frequencies or None)

def cmd_diagnose(args):
    try:
        config = load_circuit_yaml(args.config)
        circuit = AnalogCircuit(config)
        faults = parse_faults(args.fault)
        frequencies = resolve_frequencies(config, args.freq)

        acc_indices = circuit.get_accessible_indices()
        excitations = []
        for idx in acc_indices:
            J = np.zeros(circuit.num_free_nodes)
            J[idx] = 1.0
            excitations.append(J)

        blocks = calculate_measurements(circuit, excitations, faulty_elements=faults,
                                        frequencies=frequencies)

        result = diagnose_node_faults(circuit, blocks, None, args.k,
                                      top_n=args.top_n, method=args.method)

        if frequencies:
            print(f"Frequencies (Hz): {frequencies}")
        print(f"Status: {result['status']}")
        print("\nBest Candidate:")
        best = result['best']
        if best:
            print(f"  Nodes: {best['support']}")
            print(f"  Residual: {best['residual_norm']:.6e}")
            print(f"  Relative Residual: {best['relative_residual']:.6e}")
            print(f"  Cond Number: {best['condition_number']:.2f}")
        
        print("\nTop Candidates:")
        for i, cand in enumerate(result['candidates']):
            print(f"  {i+1}. Nodes: {cand['support']}, Residual: {cand['residual_norm']:.6e}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def cmd_evaluate(args):
    try:
        config = load_circuit_yaml(args.config)
        circuit = AnalogCircuit(config)
        faults = parse_faults(args.fault)
        frequencies = resolve_frequencies(config, args.freq)

        result = run_evaluation(
            circuit, faults, args.k,
            trials=args.trials, tol_percent=args.tol, noise_std=args.noise, seed=args.seed,
            frequencies=frequencies
        )
        
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Analog Fault Diagnosis MVP CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    p_test = subparsers.add_parser("testability")
    p_test.add_argument("config", help="YAML config file")
    p_test.add_argument("--k", type=int, default=1)
    
    p_diag = subparsers.add_parser("diagnose")
    p_diag.add_argument("config", help="YAML config file")
    p_diag.add_argument("--fault", help="Fault string (e.g. R3=0.1,R4=0.5)")
    p_diag.add_argument("--k", type=int, default=1)
    p_diag.add_argument("--top-n", type=int, default=5)
    p_diag.add_argument("--method", choices=["auto", "exhaustive", "omp"], default="auto",
                        help="Diagnosis strategy (default: auto - exact when small, OMP when large)")
    p_diag.add_argument("--freq", help="Comma-separated measurement frequencies in Hz "
                                       "(e.g. 1000,5000). Defaults to the circuit's frequencies.")

    p_eval = subparsers.add_parser("evaluate")
    p_eval.add_argument("config", help="YAML config file")
    p_eval.add_argument("--fault", help="Fault string (e.g. R3=0.1)")
    p_eval.add_argument("--k", type=int, default=1)
    p_eval.add_argument("--trials", type=int, default=100)
    p_eval.add_argument("--tol", type=float, default=0.0)
    p_eval.add_argument("--noise", type=float, default=0.0)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--freq", help="Comma-separated measurement frequencies in Hz "
                                       "(e.g. 1000,5000). Defaults to the circuit's frequencies.")
    
    args = parser.parse_args()
    if args.command == "testability":
        cmd_testability(args)
    elif args.command == "diagnose":
        cmd_diagnose(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
