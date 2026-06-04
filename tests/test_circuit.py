import pytest
import numpy as np

from analog_fault.schema import CircuitConfig, Element
from analog_fault.circuit import AnalogCircuit
from analog_fault.testability import check_k_node_testability
from analog_fault.simulate import calculate_delta_v
from analog_fault.diagnose import diagnose_node_faults, reconstruct_branch_faults

def test_bridge_network_regression():
    """
    analog_fault パッケージの実装が Bridge Network の完全再構築を
    正しく行えることを担保する回帰テスト。
    """
    config = CircuitConfig(
        name="bridge_test",
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
    circuit = AnalogCircuit(config)
    
    # 1. Topological Analysis
    testable_1, conns_1 = check_k_node_testability(circuit, 1)
    assert testable_1 is True
    assert conns_1[4] == 3
    
    testable_2, conns_2 = check_k_node_testability(circuit, 2)
    assert testable_2 is True
    
    # 2. Fault Simulation (R4: 1.0 -> 0.1)
    faulty_branches = {"R4": 0.1}
    excitations = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
    ]
    
    # 指摘1: テストコードから呼び出す際だけは、必ず固定シードを持たせたGeneratorインスタンスを明示的に渡す
    rng = np.random.default_rng(42)
    delta_v_ms, Z_mn = calculate_delta_v(
        circuit, excitations, faulty_elements=faulty_branches, rng=rng, tol_percent=0.0
    )
    
    # 3. Fault Diagnosis (明示的に exhaustive を指定し後方互換性を担保)
    k_diag = 2
    result = diagnose_node_faults(circuit, Z_mn, delta_v_ms, k_diag, method='exhaustive')
    
    assert result["status"] == "unique"
    best = result["best"]
    
    # 指摘2: 浮動小数点演算の丸め誤差への対応
    assert best["residual_norm"] == pytest.approx(0.0, abs=1e-12)
    
    consensus = sorted(best["support"])
    assert consensus == [1, 4]
    
    # 4. Branch Admittance Reconstruction
    branch_results = reconstruct_branch_faults(
        circuit, consensus, excitations, delta_v_ms, result
    )
    
    # 指摘2: 浮動小数点演算の丸め誤差への対応
    assert branch_results["R4"]["delta_g"] == pytest.approx(-0.9, rel=1e-5, abs=1e-8)
    assert branch_results["R1"]["delta_g"] == pytest.approx(0.0, rel=1e-5, abs=1e-8)
    assert branch_results["R5"]["delta_g"] == pytest.approx(0.0, rel=1e-5, abs=1e-8)
    assert branch_results["R6"]["delta_g"] == pytest.approx(0.0, rel=1e-5, abs=1e-8)

def _bridge_circuit():
    config = CircuitConfig(
        name="bridge_test",
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
    return AnalogCircuit(config)


def test_auto_method_is_exact_on_small_circuit():
    """
    'auto' (新デフォルト) は小規模回路で全探索を選び、Bridge Network の
    内部ノードを含む 2 ノード故障 (R4: 1->0.1, ノード {1,4}) を厳密に特定できる。
    貪欲な OMP 単体ではこの対称回路の内部ノード故障を取りこぼすため、
    'auto' による自動フォールバックが結果の信頼性を担保することを確認する。
    """
    circuit = _bridge_circuit()
    excitations = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
    ]
    delta_v_ms, Z_mn = calculate_delta_v(
        circuit, excitations, faulty_elements={"R4": 0.1}, tol_percent=0.0
    )

    result = diagnose_node_faults(circuit, Z_mn, delta_v_ms, max_faults=2)  # method='auto'

    assert sorted(result["best"]["support"]) == [1, 4]
    assert result["best"]["relative_residual"] == pytest.approx(0.0, abs=1e-9)


def test_omp_does_not_overselect_single_node_fault():
    """
    真の故障が 1 ノード (R1: 1->0.1, ノード {1}) のとき、max_faults=2 を
    指定しても OMP が 2 ノードを誤って報告しない (過剰選択しない) ことを担保する。
    """
    circuit = _bridge_circuit()
    excitations = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
    ]
    delta_v_ms, Z_mn = calculate_delta_v(
        circuit, excitations, faulty_elements={"R1": 0.1}, tol_percent=0.0
    )

    result = diagnose_node_faults(circuit, Z_mn, delta_v_ms, max_faults=2, method='omp')

    # ノード 1 のみが故障。max_faults=2 でも 1 ノードに収束すべき。
    assert sorted(result["best"]["support"]) == [1]


def test_ladder_network_regression():
    """
    analog_fault パッケージの実装が Ladder Network における再構築の限界を
    正しく処理できることを担保する回帰テスト。
    """
    config = CircuitConfig(
        name="ladder_test",
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
    circuit = AnalogCircuit(config)
    
    # 1. Topological Analysis
    testable_1, conns_1 = check_k_node_testability(circuit, 1)
    assert testable_1 is True
    assert conns_1[4] == 3 and conns_1[5] == 3
    
    # 2. Fault Simulation (R3: 1.5 -> 0.1)
    faulty_branches = {"R3": 0.1}
    excitations = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    ]
    
    # 指摘1: テストコードから呼び出す際だけは、必ず固定シードを持たせたGeneratorインスタンスを明示的に渡す
    rng = np.random.default_rng(42)
    # tol_percent=2.0 にしてランダム性をテストに組み込む
    delta_v_ms, Z_mn = calculate_delta_v(
        circuit, excitations, faulty_elements=faulty_branches, rng=rng, tol_percent=2.0
    )
    
    # 3. Fault Diagnosis (明示的に exhaustive を指定し後方互換性を担保)
    k_diag = 1
    result = diagnose_node_faults(circuit, Z_mn, delta_v_ms, k_diag, method='exhaustive')
    
    best = result["best"]
    consensus = sorted(best["support"])
    assert consensus == [4]
