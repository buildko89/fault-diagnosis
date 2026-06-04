"""
AC / 複素アドミタンス対応 (F) のテスト。
Phase 2: element_admittance と build_matrices(omega=...) の複素行列構築を検証。
"""
import numpy as np
import pytest

from analog_fault.schema import CircuitConfig, Element
from analog_fault.circuit import AnalogCircuit, element_admittance


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
