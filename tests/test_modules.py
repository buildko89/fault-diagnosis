"""
fault の補助モジュール（schema 検証 / YAML ロード / testability の偽判定 /
evaluate / Ridge による枝再構築）に対する回帰テスト。
"""
import numpy as np
import pytest

from fault.schema import (
    CircuitConfig,
    Element,
    load_circuit_yaml,
    validate_config,
)
from fault.circuit import Circuit
from fault.testability import check_k_node_testability
from fault.simulate import calculate_delta_v
from fault.diagnose import diagnose_node_faults, reconstruct_branch_faults
from fault.evaluate import run_evaluation


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


# --------------------------------------------------------------------------
# schema: validate_config の異常系
# --------------------------------------------------------------------------
def test_validate_rejects_reference_not_in_nodes():
    cfg = CircuitConfig("c", reference=9, nodes=[0, 1], accessible=[1],
                        elements=[Element("R1", "R", 0, 1, 1.0)])
    with pytest.raises(ValueError, match="Reference node"):
        validate_config(cfg)


def test_validate_rejects_no_accessible_besides_reference():
    cfg = CircuitConfig("c", reference=0, nodes=[0, 1], accessible=[0],
                        elements=[Element("R1", "R", 0, 1, 1.0)])
    with pytest.raises(ValueError, match="other than the reference"):
        validate_config(cfg)


def test_validate_rejects_duplicate_element_names():
    cfg = CircuitConfig("c", reference=0, nodes=[0, 1, 2], accessible=[1],
                        elements=[Element("R1", "R", 0, 1, 1.0),
                                  Element("R1", "R", 1, 2, 1.0)])
    with pytest.raises(ValueError, match="unique"):
        validate_config(cfg)


def test_validate_rejects_unsupported_type():
    cfg = CircuitConfig("c", reference=0, nodes=[0, 1], accessible=[1],
                        elements=[Element("X1", "X", 0, 1, 1.0)])
    with pytest.raises(ValueError, match="unsupported type"):
        validate_config(cfg)


def test_validate_accepts_rcl_types_with_frequencies():
    cfg = CircuitConfig("rcl", reference=0, nodes=[0, 1, 2], accessible=[1, 2],
                        elements=[Element("R1", "R", 1, 0, 1.0),
                                  Element("C1", "C", 1, 2, 1e-6),
                                  Element("L1", "L", 2, 0, 1e-3)],
                        frequencies=[1000.0, 5000.0])
    validate_config(cfg)  # should not raise


def test_validate_rejects_reactive_without_frequency():
    cfg = CircuitConfig("rc", reference=0, nodes=[0, 1, 2], accessible=[1, 2],
                        elements=[Element("R1", "R", 1, 0, 1.0),
                                  Element("C1", "C", 1, 2, 1e-6)])
    with pytest.raises(ValueError, match="require at least one positive frequency"):
        validate_config(cfg)


def test_validate_rejects_nonpositive_frequency():
    cfg = CircuitConfig("rc", reference=0, nodes=[0, 1, 2], accessible=[1, 2],
                        elements=[Element("R1", "R", 1, 0, 1.0),
                                  Element("C1", "C", 1, 2, 1e-6)],
                        frequencies=[0.0])
    with pytest.raises(ValueError, match="Frequencies must be positive"):
        validate_config(cfg)


def test_load_circuit_yaml_reads_frequencies(tmp_path):
    yaml_text = """
name: rc
reference: 0
nodes: [0, 1, 2]
accessible: [1, 2]
frequencies: [1000.0, 2000.0]
elements:
  - {name: R1, type: R, n1: 1, n2: 0, value: 1.0}
  - {name: C1, type: C, n1: 1, n2: 2, value: 1.0e-6}
"""
    p = tmp_path / "rc.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_circuit_yaml(str(p))
    assert cfg.frequencies == [1000.0, 2000.0]
    assert cfg.elements[1] == Element("C1", "C", 1, 2, 1.0e-6)


def test_validate_rejects_nonpositive_value():
    cfg = CircuitConfig("c", reference=0, nodes=[0, 1], accessible=[1],
                        elements=[Element("R1", "R", 0, 1, 0.0)])
    with pytest.raises(ValueError, match="positive"):
        validate_config(cfg)


def test_validate_rejects_self_loop_element():
    cfg = CircuitConfig("c", reference=0, nodes=[0, 1], accessible=[1],
                        elements=[Element("R1", "R", 1, 1, 1.0)])
    with pytest.raises(ValueError, match="identical nodes"):
        validate_config(cfg)


# --------------------------------------------------------------------------
# schema: YAML ロードの往復
# --------------------------------------------------------------------------
def test_load_circuit_yaml_roundtrip(tmp_path):
    yaml_text = """
name: tiny
reference: 0
nodes: [0, 1, 2]
accessible: [1, 2]
elements:
  - {name: R1, type: R, n1: 1, n2: 0, value: 2.0}
  - {name: R2, type: R, n1: 1, n2: 2, value: 3.0}
"""
    p = tmp_path / "tiny.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    cfg = load_circuit_yaml(str(p))

    assert cfg.name == "tiny"
    assert cfg.reference == 0
    assert cfg.accessible == [1, 2]
    assert len(cfg.elements) == 2
    assert cfg.elements[0] == Element("R1", "R", 1, 0, 2.0)


def test_load_circuit_yaml_validates(tmp_path):
    # reference (5) が nodes に存在しない -> ロード時に検証エラー
    bad = """
name: bad
reference: 5
nodes: [0, 1]
accessible: [1]
elements:
  - {name: R1, type: R, n1: 0, n2: 1, value: 1.0}
"""
    p = tmp_path / "bad.yaml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError):
        load_circuit_yaml(str(p))


# --------------------------------------------------------------------------
# testability: 「偽」判定（独立パス不足）
# --------------------------------------------------------------------------
def test_testability_false_when_paths_insufficient():
    # 直列チェーン: ノード2,3 は接地/アクセス可能ノードへの独立パスが 1 本のみ。
    cfg = CircuitConfig("chain", reference=0, nodes=[0, 1, 2, 3], accessible=[1],
                        elements=[Element("R1", "R", 1, 0, 1.0),
                                  Element("R2", "R", 1, 2, 1.0),
                                  Element("R3", "R", 2, 3, 1.0)])
    circuit = Circuit(cfg)

    testable, conns = check_k_node_testability(circuit, k=1)
    assert testable is False
    assert conns[2] == 1
    assert conns[3] == 1


# --------------------------------------------------------------------------
# evaluate: 故障なし誤差・精度集計（auto 既定で厳密）
# --------------------------------------------------------------------------
def test_run_evaluation_recovers_single_accessible_fault():
    circuit = Circuit(_bridge_config())
    # R1 (ノード1, アクセス可能) の単一故障 -> auto(=exhaustive) で必ず特定できる。
    result = run_evaluation(circuit, {"R1": 0.1}, max_faults=1,
                            trials=5, tol_percent=0.0, noise_std=0.0, seed=1)

    assert result["correct_nodes"] == [1]
    assert result["top1_accuracy"] == 1.0
    assert result["top3_accuracy"] == 1.0


def test_run_evaluation_warns_when_faults_exceed_capacity():
    circuit = Circuit(_bridge_config())
    # 実故障ノード {1,2} に対し max_faults=1 は不足 -> 警告が出る。
    with pytest.warns(UserWarning, match="exceeds max_faults"):
        run_evaluation(circuit, {"R1": 0.1, "R2": 0.1}, max_faults=1,
                       trials=2, tol_percent=0.0, noise_std=0.0, seed=1)


# --------------------------------------------------------------------------
# diagnose: Ridge による枝アドミタンス再構築
# --------------------------------------------------------------------------
def test_reconstruct_branch_faults_ridge_identifies_faulty_branch():
    circuit = Circuit(_bridge_config())
    excitations = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
    ]
    delta_v_ms, Z_mn = calculate_delta_v(
        circuit, excitations, faulty_elements={"R4": 0.1}, tol_percent=0.0
    )
    result = diagnose_node_faults(circuit, Z_mn, delta_v_ms, max_faults=2,
                                  method="exhaustive")
    support = sorted(result["best"]["support"])

    branch = reconstruct_branch_faults(
        circuit, support, excitations, delta_v_ms, result,
        method="ridge", alpha=1.0,
    )

    assert set(branch.keys()) == {"R1", "R4", "R5", "R6"}
    # 実際に故障した R4 が最も負方向（コンダクタンス低下）に推定される。
    most_negative = min(branch, key=lambda k: branch[k]["delta_g"])
    assert most_negative == "R4"
    assert branch["R4"]["delta_g"] < 0.0
