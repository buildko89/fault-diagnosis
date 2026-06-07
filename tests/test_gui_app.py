"""Headless smoke test of the Streamlit app via AppTest.

Skipped automatically when streamlit isn't installed (the GUI is an optional
``[gui]`` extra), so the core test run never depends on it.
"""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fault_gui", "app.py")


def test_app_loads_and_testability_runs():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    # 4 main tabs (+ inner tabs); at least the 4 top-level ones exist.
    assert len(at.tabs) >= 4

    # The default example (examples/bridge.yaml) is testable at k=2.
    at.button(key="run_testability").click().run()
    assert not at.exception
    assert any("Testable: True" in s.value for s in at.success)


def test_evaluate_without_fault_shows_info():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button(key="run_evaluate").click().run()
    assert not at.exception
    assert any("素子" in i.value for i in at.info)
