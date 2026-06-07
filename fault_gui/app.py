"""Streamlit entry point for the fault-diagnosis GUI.

Run with::

    streamlit run fault_gui/app.py
    # or, once installed:  fault-gui

The page reads the circuit + common parameters from the sidebar and dispatches
each tab to ``fault_gui.views``. All computation goes through ``fault.service``.
"""
import os

import streamlit as st

from fault import service
from fault_gui.views import (
    render_circuit,
    render_testability,
    render_diagnose,
    render_evaluate,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")


def _list_examples():
    if not os.path.isdir(EXAMPLES_DIR):
        return []
    return sorted(f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".yaml"))


def _load_from_sidebar():
    """Render the circuit-source controls and return a Circuit or None."""
    st.sidebar.header("回路ソース")
    mode = st.sidebar.radio(
        "読込方法", ["examples から選択", "YAML アップロード", "パス入力"],
        key="src_mode",
    )

    try:
        if mode == "examples から選択":
            examples = _list_examples()
            if not examples:
                st.sidebar.warning("examples/ に YAML が見つかりません。")
                return None
            name = st.sidebar.selectbox("ファイル", examples, key="src_example")
            return service.load_circuit(os.path.join(EXAMPLES_DIR, name))

        if mode == "YAML アップロード":
            up = st.sidebar.file_uploader("YAML ファイル", type=["yaml", "yml"])
            if up is None:
                return None
            return service.load_circuit(up.getvalue().decode("utf-8"), is_text=True)

        path = st.sidebar.text_input("YAML パス", key="src_path")
        if not path:
            return None
        return service.load_circuit(path)
    except Exception as e:  # noqa: BLE001 - surface load/validation errors in UI
        st.sidebar.error(f"読込エラー: {e}")
        return None


def render():
    st.set_page_config(page_title="Fault Diagnosis GUI", layout="wide")
    st.title("Circuit Fault Diagnosis — GUI")

    circuit = _load_from_sidebar()

    st.sidebar.header("共通パラメータ")
    k = st.sidebar.number_input("k (最大故障数)", min_value=1, max_value=20, value=2, step=1)
    freq_text = st.sidebar.text_input(
        "frequencies [Hz]（カンマ区切り、空欄は回路設定/DC）", key="freq_text",
        help="例: 1000,5000。空欄なら回路の frequencies、無ければ DC。",
    )

    if circuit is None:
        st.info("← サイドバーから回路を読み込んでください。")
        return

    try:
        frequencies = service.resolve_frequencies(circuit.config, freq_text or None)
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"周波数エラー: {e}")
        frequencies = None

    tab_c, tab_t, tab_d, tab_e = st.tabs(
        ["Circuit", "Testability", "Diagnose", "Evaluate"]
    )
    with tab_c:
        render_circuit(circuit)
    with tab_t:
        render_testability(circuit, int(k))
    with tab_d:
        render_diagnose(circuit, int(k), frequencies)
    with tab_e:
        render_evaluate(circuit, int(k), frequencies)


# Streamlit executes this script as "__main__"; importing the module (e.g. in
# tests) must NOT build the page.
if __name__ == "__main__":
    render()
