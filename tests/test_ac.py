"""
AC / 複素アドミタンス対応 (F) のテスト。
Phase 2: element_admittance と build_matrices(omega=...) の複素行列構築を検証。
"""
import numpy as np
import pytest

from analog_fault.schema import CircuitConfig, Element
from analog_fault.circuit import AnalogCircuit, element_admittance
from analog_fault.simulate import calculate_measurements, calculate_delta_v
from analog_fault.diagnose import diagnose_node_faults


# --------------------------------------------------------------------------
# element_admittance: 各素子の複素アドミタンス
# --------------------------------------------------------------------------
def test_element_admittance_resistor_is_real_and_freq_independent():
    assert element_admittance("R", 2.0, omega=None) == 2.0
    assert element_admittance("R", 2.0, omega=1000.0) == 2.0


def test_element_admittance_capacitor():
    # Y_C = j*omega*C
    assert element_admittance("C", 1e-6, omega=2000.0) == pytest.approx(1j * 2000.0 * 1e-6)


def test_element_admittance_inductor():
    # Y_L = 1/(j*omega*L) = -j/(omega*L)
    y = element_admittance("L", 0.5, omega=2.0)
    assert y == pytest.approx(1.0 / (1j * 2.0 * 0.5))
    assert y == pytest.approx(-1j)  # 1/(j*1) = -j


def test_element_admittance_reactive_at_dc_raises():
    with pytest.raises(ValueError, match="requires an angular frequency"):
        element_admittance("C", 1e-6, omega=None)


# --------------------------------------------------------------------------
# build_matrices: 複素 Y(omega) 構築
# --------------------------------------------------------------------------
def _rc_shunt_circuit():
    # node1 to ground via R (G=1.0) and C (C=1.0), measured at node1.
    cfg = CircuitConfig(
        name="rc",
        reference=0,
        nodes=[0, 1],
        accessible=[1],
        elements=[
            Element("R1", "R", 1, 0, 1.0),
            Element("C1", "C", 1, 0, 1.0),
        ],
        frequencies=[1.0],
    )
    return AnalogCircuit(cfg)


def test_build_matrices_complex_Y_for_rc():
    circuit = _rc_shunt_circuit()
    omega = 2.0
    A, Yb, Y = circuit.build_matrices(omega=omega)

    Yd = Y.toarray()
    assert Yd.dtype == np.complex128
    assert Yd.shape == (1, 1)
    # Y = G + j*omega*C = 1 + j*2*1
    assert Yd[0, 0] == pytest.approx(1.0 + 2.0j)


def test_build_matrices_dc_path_stays_real():
    # R-only circuit with omega=None -> real matrices (backward compatible).
    cfg = CircuitConfig("r", reference=0, nodes=[0, 1], accessible=[1],
                        elements=[Element("R1", "R", 1, 0, 1.0)])
    circuit = AnalogCircuit(cfg)
    A, Yb, Y = circuit.build_matrices()  # omega=None
    assert Y.toarray().dtype == np.float64
    assert Y.toarray()[0, 0] == pytest.approx(1.0)


def test_build_matrices_faulty_value_is_component_value():
    # Faulty capacitance C: 1.0 -> 3.0 should change Y imag part to j*omega*3.
    circuit = _rc_shunt_circuit()
    omega = 2.0
    _, _, Y = circuit.build_matrices(faulty_elements={"C1": 3.0}, omega=omega)
    assert Y.toarray()[0, 0] == pytest.approx(1.0 + 6.0j)


# --------------------------------------------------------------------------
# calculate_measurements: 多周波数の複素シミュレーション
# --------------------------------------------------------------------------
def test_calculate_measurements_rc_matches_analytic():
    # 1 free node (node1), shunt G=1 and C=1 to ground; inject 1 A at node1.
    circuit = _rc_shunt_circuit()
    f = 1.0
    omega = 2.0 * np.pi * f
    G, C, Cf = 1.0, 1.0, 2.0
    excitations = [np.array([1.0])]

    blocks = calculate_measurements(
        circuit, excitations, faulty_elements={"C1": Cf}, frequencies=[f]
    )
    assert len(blocks) == 1
    blk = blocks[0]
    assert blk.omega == pytest.approx(omega)

    # Z_mn = 1 / Y_nom = 1/(G + j*omega*C)
    assert blk.Z_mn.shape == (1, 1)
    assert blk.Z_mn[0, 0] == pytest.approx(1.0 / (G + 1j * omega * C))

    # ΔV = 1/(G + j*omega*Cf) - 1/(G + j*omega*C)
    expected = 1.0 / (G + 1j * omega * Cf) - 1.0 / (G + 1j * omega * C)
    assert blk.delta_v_ms[0][0] == pytest.approx(expected)


def test_calculate_measurements_multiple_frequencies():
    circuit = _rc_shunt_circuit()
    freqs = [10.0, 100.0, 1000.0]
    blocks = calculate_measurements(
        circuit, [np.array([1.0])], faulty_elements={"C1": 2.0}, frequencies=freqs
    )
    assert len(blocks) == 3
    assert [b.omega for b in blocks] == pytest.approx([2 * np.pi * f for f in freqs])
    # Each block carries complex measurements.
    for b in blocks:
        assert np.iscomplexobj(b.Z_mn)
        assert np.iscomplexobj(b.delta_v_ms[0])


def test_calculate_delta_v_wrapper_matches_single_block():
    # The DC wrapper must return exactly the single-block result.
    cfg = CircuitConfig("r", reference=0, nodes=[0, 1, 2], accessible=[1, 2],
                        elements=[Element("R1", "R", 1, 0, 1.0),
                                  Element("R2", "R", 1, 2, 1.0),
                                  Element("R3", "R", 2, 0, 1.0)])
    circuit = AnalogCircuit(cfg)
    exc = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    dv, Z = calculate_delta_v(circuit, exc, faulty_elements={"R2": 0.1})
    block = calculate_measurements(circuit, exc, faulty_elements={"R2": 0.1})[0]

    assert np.allclose(Z, block.Z_mn)
    assert block.omega is None
    for a, b in zip(dv, block.delta_v_ms):
        assert np.allclose(a, b)


# --------------------------------------------------------------------------
# diagnose: 複素・多周波数の故障診断（MeasurementBlock のリストを受理）
# --------------------------------------------------------------------------
def _ac_bridge_with_cap():
    # Bridge topology with one capacitor branch (C4 between node1 and node4).
    cfg = CircuitConfig(
        name="ac_bridge",
        reference=0,
        nodes=[0, 1, 2, 3, 4],
        accessible=[1, 2, 3],
        elements=[
            Element("R1", "R", 1, 0, 1.0),
            Element("R2", "R", 2, 0, 1.0),
            Element("R3", "R", 3, 0, 1.0),
            Element("C4", "C", 1, 4, 1.0e-3),
            Element("R5", "R", 2, 4, 1.0),
            Element("R6", "R", 3, 4, 1.0),
        ],
        frequencies=[1000.0, 5000.0],
    )
    return AnalogCircuit(cfg)


def _unit_excitations(circuit):
    return [np.eye(circuit.num_free_nodes)[idx]
            for idx in circuit.get_accessible_indices()]


def test_ac_diagnosis_single_frequency_recovers_fault():
    circuit = _ac_bridge_with_cap()
    exc = _unit_excitations(circuit)
    blocks = calculate_measurements(
        circuit, exc, faulty_elements={"C4": 0.5e-3}, frequencies=[1000.0]
    )
    # Pass blocks list as the data argument; delta_v_ms is ignored.
    result = diagnose_node_faults(circuit, blocks, None, max_faults=2)  # auto -> exhaustive
    assert sorted(result["best"]["support"]) == [1, 4]
    assert result["best"]["relative_residual"] == pytest.approx(0.0, abs=1e-9)


def test_ac_diagnosis_multi_frequency_recovers_fault():
    circuit = _ac_bridge_with_cap()
    exc = _unit_excitations(circuit)
    blocks = calculate_measurements(
        circuit, exc, faulty_elements={"C4": 0.5e-3}, frequencies=[1000.0, 5000.0]
    )
    assert len(blocks) == 2
    result = diagnose_node_faults(circuit, blocks, None, max_faults=2)
    assert sorted(result["best"]["support"]) == [1, 4]
    assert result["best"]["relative_residual"] == pytest.approx(0.0, abs=1e-9)


def test_diagnose_blocks_api_runs_with_omp():
    # The blocks API must also work via the explicit OMP path (returns a support).
    circuit = _ac_bridge_with_cap()
    exc = _unit_excitations(circuit)
    blocks = calculate_measurements(
        circuit, exc, faulty_elements={"C4": 0.5e-3}, frequencies=[1000.0, 5000.0]
    )
    result = diagnose_node_faults(circuit, blocks, None, max_faults=2, method="omp")
    assert result["best"] is not None
    assert 1 <= len(result["best"]["support"]) <= 2
