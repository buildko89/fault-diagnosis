"""Cached compute wrappers around fault.service.

Streamlit's ``@st.cache_data`` keys on the function arguments, so these take the
circuit's YAML *text* (hashable) plus primitive params and rebuild the Circuit
inside. Identical inputs then reuse the previous result instead of recomputing
(useful when flipping tabs or tweaking unrelated widgets).
"""
from typing import Optional, Tuple

import streamlit as st

from fault import service


def _circuit(yaml_text: str):
    return service.load_circuit(yaml_text, is_text=True)


@st.cache_data(show_spinner=False)
def testability(yaml_text: str, k: int):
    return service.run_testability(_circuit(yaml_text), k)


@st.cache_data(show_spinner=False)
def diagnose(
    yaml_text: str,
    faults_items: Tuple[Tuple[str, float], ...],
    k: int,
    top_n: int,
    method: str,
    frequencies: Optional[Tuple[float, ...]],
    reconstruct: bool,
):
    return service.run_diagnose(
        _circuit(yaml_text), dict(faults_items), k,
        top_n=top_n, method=method,
        frequencies=list(frequencies) if frequencies else None,
        reconstruct=reconstruct,
    )


@st.cache_data(show_spinner=False)
def evaluate(
    yaml_text: str,
    faults_items: Tuple[Tuple[str, float], ...],
    k: int,
    trials: int,
    tol: float,
    noise: float,
    seed: int,
    frequencies: Optional[Tuple[float, ...]],
):
    return service.run_evaluate(
        _circuit(yaml_text), dict(faults_items), k,
        trials=trials, tol=tol, noise=noise, seed=seed,
        frequencies=list(frequencies) if frequencies else None,
    )
