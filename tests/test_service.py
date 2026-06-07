"""Tests for the framework-agnostic service layer (fault/service.py).

These exercise the same flows the CLI and GUI call, asserting the exact-search
results so the front-ends share verified behavior.
"""
import os

import numpy as np
import pytest

from fault.schema import CircuitConfig, Element
from fault.circuit import Circuit
from fault import service


EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


def _bridge_config():
    return CircuitConfig(
        name="bridge",
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
        ],
    )


def _rc_config():
    return CircuitConfig(
        name="rc", reference=0, nodes=[0, 1, 2], accessible=[1, 2],
        frequencies=[1000.0, 5000.0],
        elements=[
            Element("R1", "R", 1, 0, 1.0),
            Element("C1", "C", 1, 2, 1.0e-3),
        ],
    )


# --------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------
def test_parse_faults_basic():
    assert service.parse_faults("R3=0.1,R4=0.5") == {"R3": 0.1, "R4": 0.5}


def test_parse_faults_empty_returns_empty_dict():
    assert service.parse_faults("") == {}
    assert service.parse_faults(None) == {}


def test_parse_faults_rejects_malformed():
    with pytest.raises(ValueError, match="Invalid fault format"):
        service.parse_faults("R3")


def test_parse_freqs():
    assert service.parse_freqs("1000,5000") == [1000.0, 5000.0]
    assert service.parse_freqs("") is None


def test_resolve_frequencies_prefers_arg_then_config():
    cfg = _rc_config()
    assert service.resolve_frequencies(cfg, "2000") == [2000.0]
    assert service.resolve_frequencies(cfg, None) == [1000.0, 5000.0]


# --------------------------------------------------------------------------
# loading & summary
# --------------------------------------------------------------------------
def test_load_circuit_from_path():
    circuit = service.load_circuit(os.path.join(EXAMPLES_DIR, "bridge.yaml"))
    assert isinstance(circuit, Circuit)
    assert circuit.config.name == "bridge"


def test_load_circuit_from_text():
    text = """
name: tiny
reference: 0
nodes: [0, 1, 2]
accessible: [1, 2]
elements:
  - {name: R1, type: R, n1: 1, n2: 0, value: 2.0}
  - {name: R2, type: R, n1: 1, n2: 2, value: 3.0}
"""
    circuit = service.load_circuit(text)
    assert circuit.config.name == "tiny"
    assert len(circuit.config.elements) == 2


def test_circuit_summary_shape():
    circuit = Circuit(_bridge_config())
    summary = service.circuit_summary(circuit)
    assert summary["name"] == "bridge"
    assert summary["accessible"] == [1, 2, 3]
    assert summary["inaccessible"] == [4]
    assert summary["num_elements"] == 6
    assert summary["frequencies"] is None
    assert {"name", "type", "n1", "n2", "value"} <= set(summary["elements"][0])


# --------------------------------------------------------------------------
# testability
# --------------------------------------------------------------------------
def test_run_testability_chain_is_false():
    cfg = CircuitConfig("chain", reference=0, nodes=[0, 1, 2, 3], accessible=[1],
                        elements=[Element("R1", "R", 1, 0, 1.0),
                                  Element("R2", "R", 1, 2, 1.0),
                                  Element("R3", "R", 2, 3, 1.0)])
    res = service.run_testability(Circuit(cfg), k=1)
    assert res["testable"] is False
    assert res["connectivities"][2] == 1
    assert res["connectivities"][3] == 1


# --------------------------------------------------------------------------
# diagnose
# --------------------------------------------------------------------------
def test_run_diagnose_dc_identifies_accessible_fault():
    circuit = Circuit(_bridge_config())
    result = service.run_diagnose(circuit, {"R1": 0.1}, k=1, method="exhaustive")
    assert result["best"]["support"] == [1]
    assert result["frequencies"] is None


def test_run_diagnose_with_reconstruct_dc():
    circuit = Circuit(_bridge_config())
    result = service.run_diagnose(circuit, {"R4": 0.1}, k=2,
                                  method="exhaustive", reconstruct=True)
    branch = result["branch"]
    assert "R4" in branch
    # The actually-faulted R4 has the most negative conductance deviation.
    most_negative = min(branch, key=lambda n: branch[n]["delta_g"])
    assert most_negative == "R4"


def test_run_diagnose_ac_classifies_capacitor():
    circuit = Circuit(_rc_config())
    result = service.run_diagnose(
        circuit, {"C1": 0.5e-3}, k=2, method="exhaustive",
        frequencies=[1000.0, 5000.0], reconstruct=True,
    )
    assert result["frequencies"] == [1000.0, 5000.0]
    assert result["branch"]["C1"]["classification"] == "C"


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------
def test_run_evaluate_recovers_single_fault():
    circuit = Circuit(_bridge_config())
    result = service.run_evaluate(circuit, {"R1": 0.1}, k=1,
                                  trials=5, tol=0.0, noise=0.0, seed=1)
    assert result["correct_nodes"] == [1]
    assert result["top1_accuracy"] == 1.0
    assert result["top3_accuracy"] == 1.0
