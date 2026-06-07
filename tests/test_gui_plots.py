"""Smoke tests for the GUI figure builders (matplotlib only, no Streamlit).

These guard against import/rendering regressions in fault_gui.plots without
needing a running Streamlit server.
"""
import numpy as np
from matplotlib.figure import Figure

from fault.schema import CircuitConfig, Element
from fault.circuit import Circuit
from fault_gui.plots import topology_figure, delta_v_figure, accuracy_sweep_figure


def _bridge():
    return Circuit(CircuitConfig(
        name="bridge", reference=0, nodes=[0, 1, 2, 3, 4], accessible=[1, 2, 3],
        elements=[
            Element("R1", "R", 1, 0, 1.0), Element("R2", "R", 2, 0, 1.0),
            Element("R3", "R", 3, 0, 1.0), Element("R4", "R", 1, 4, 1.0),
            Element("R5", "R", 2, 4, 1.0), Element("R6", "R", 3, 4, 1.0),
        ],
    ))


def test_topology_figure_returns_figure():
    fig = topology_figure(_bridge(), faulty_nodes=[1, 4])
    assert isinstance(fig, Figure)


def test_delta_v_figure_dc():
    fig = delta_v_figure(_bridge(), np.array([0.1, -0.2, 0.05]))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1  # single panel for real ΔV


def test_delta_v_figure_ac_has_two_panels():
    dv = np.array([0.1 + 0.2j, -0.2 + 0.1j, 0.05 - 0.05j])
    fig = delta_v_figure(_bridge(), dv)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2  # magnitude + phase


def test_accuracy_sweep_figure():
    fig = accuracy_sweep_figure([0.0, 1.0, 2.0], [100.0, 80.0, 60.0],
                                xlabel="tolerance [%]")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
