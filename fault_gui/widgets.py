"""Reusable Streamlit input widgets."""
from typing import Dict

import pandas as pd
import streamlit as st


def fault_input(elements, key_prefix: str = "fault") -> Dict[str, float]:
    """Render an editable fault table and return ``{name: value}``.

    Uses ``st.data_editor`` with dynamic rows so faults can be added/removed
    freely. The element column is a dropdown (no free typing -> no name typos,
    a pain point of the CLI's ``--fault R3=0.1`` form). A blank value falls back
    to the element's nominal value. Nominal values are shown for reference.
    """
    names = [el.name for el in elements]
    nominal = {el.name: el.value for el in elements}

    with st.expander("素子の公称値（参考）"):
        st.dataframe(
            [{"element": el.name, "type": el.type, "nominal": el.value} for el in elements],
            width="stretch", hide_index=True,
        )

    base = pd.DataFrame({
        "element": pd.Series(dtype="object"),
        "faulty_value": pd.Series(dtype="float"),
    })
    edited = st.data_editor(
        base,
        num_rows="dynamic",
        key=f"{key_prefix}_editor",
        width="stretch",
        hide_index=True,
        column_config={
            "element": st.column_config.SelectboxColumn(
                "素子", options=names, required=True,
                help="故障させる素子を選択（行は + で追加 / 選択して削除）。",
            ),
            "faulty_value": st.column_config.NumberColumn(
                "故障値", format="%.6g",
                help="空欄なら公称値（＝偏差なし）。",
            ),
        },
    )

    faults: Dict[str, float] = {}
    for _, row in edited.iterrows():
        name = row["element"]
        if name is None or (isinstance(name, float) and pd.isna(name)) or name == "":
            continue
        val = row["faulty_value"]
        if val is None or pd.isna(val):
            val = nominal.get(name)
        if val is not None:
            faults[name] = float(val)
    return faults
