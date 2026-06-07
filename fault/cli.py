import argparse
import sys
import json

from .service import (
    parse_faults,
    resolve_frequencies,
    load_circuit,
    run_testability,
    run_diagnose,
    run_evaluate,
)


def cmd_testability(args):
    try:
        circuit = load_circuit(args.config)
        res = run_testability(circuit, args.k)
        print(f"Config: {res['name']}")
        print(f"K: {res['k']}")
        print(f"Testable: {res['testable']}")
        print(f"Connectivities: {res['connectivities']}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_diagnose(args):
    try:
        circuit = load_circuit(args.config)
        faults = parse_faults(args.fault)
        frequencies = resolve_frequencies(circuit.config, args.freq)

        result = run_diagnose(circuit, faults, args.k,
                              top_n=args.top_n, method=args.method,
                              frequencies=frequencies)

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
        circuit = load_circuit(args.config)
        faults = parse_faults(args.fault)
        frequencies = resolve_frequencies(circuit.config, args.freq)

        result = run_evaluate(
            circuit, faults, args.k,
            trials=args.trials, tol=args.tol, noise=args.noise, seed=args.seed,
            frequencies=frequencies
        )

        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Fault Diagnosis MVP CLI")
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
