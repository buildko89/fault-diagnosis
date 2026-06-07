"""Regenerate the example figures embedded in EVALUATION.md.

Run from the repo root:  python assets/eval/generate_figures.py
Uses the real diagnosis/evaluation code so the figures stay authentic.
"""
import os

import matplotlib.pyplot as plt

from fault import service
from fault_gui.plots import topology_figure, delta_v_figure, accuracy_sweep_figure

HERE = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    # 1. Topology with a diagnosed accessible fault (node 1) highlighted.
    bridge = service.load_circuit(os.path.join("examples", "bridge.yaml"))
    res = service.run_diagnose(bridge, {"R1": 0.1}, k=1, method="exhaustive")
    save(topology_figure(bridge, res["best"]["support"]), "topology_bridge.png")

    # 2a. DC voltage deviation (single panel) for the same fault.
    save(delta_v_figure(bridge, res["delta_v_m"]), "delta_v_bridge_dc.png")

    # 2b. AC voltage deviation (magnitude + phase) for an RC bridge cap fault.
    rc = service.load_circuit(os.path.join("examples", "rc_bridge.yaml"))
    res_ac = service.run_diagnose(rc, {"C4": 0.5e-3}, k=2, method="exhaustive",
                                  frequencies=[1000.0, 5000.0])
    save(delta_v_figure(rc, res_ac["delta_v_m"]), "delta_v_rc_ac.png")

    # 3. Monte Carlo accuracy vs measurement noise (degradation curve).
    noises = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
    top1 = []
    for n in noises:
        r = service.run_evaluate(bridge, {"R4": 0.3}, k=2, trials=200,
                                 tol=0.0, noise=n, seed=1)
        top1.append(r["top1_accuracy"] * 100.0)
    save(accuracy_sweep_figure(noises, top1, xlabel="noise std"), "sweep_noise.png")


if __name__ == "__main__":
    main()
