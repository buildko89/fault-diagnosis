"""Reusable Streamlit input widgets."""
from typing import Dict

import streamlit as st


def fault_input(elements, key_prefix: str = "fault") -> Dict[str, float]:
    """Render a fault picker: choose elements, then set each faulty value.

    Element names come from a dropdown (no free typing -> no name typos, which
    was a pain point of the CLI ``--fault R3=0.1`` form). The default value for
    each picked element is its nominal value. Returns ``{name: value}``.
    """
    by_name = {el.name: el for el in elements}
    chosen = st.multiselect(
        "故障させる素子", options=list(by_name.keys()), key=f"{key_prefix}_sel",
        help="選択した素子の値を変更し、その故障を診断します。",
    )

    faults: Dict[str, float] = {}
    for name in chosen:
        el = by_name[name]
        unit = {"R": "G [S]", "C": "C [F]", "L": "L [H]"}.get(el.type, "value")
        faults[name] = st.number_input(
            f"{name} ({el.type}) — 故障値 {unit}",
            value=float(el.value), format="%.6g", key=f"{key_prefix}_val_{name}",
        )
    return faults
