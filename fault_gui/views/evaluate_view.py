"""Evaluate tab: Monte Carlo accuracy under tolerance + noise."""
import streamlit as st

from fault import service
from ..widgets import fault_input


def render_evaluate(circuit, k: int, frequencies):
    st.subheader("モンテカルロ評価")
    st.caption(f"共通パラメータ k = {k} / frequencies = {frequencies or 'DC'}（サイドバーで変更）")

    faults = fault_input(circuit.config.elements, key_prefix="eval")

    c1, c2, c3, c4 = st.columns(4)
    trials = c1.number_input("trials", min_value=1, max_value=10000, value=100, step=10)
    tol = c2.number_input("tolerance [%]", min_value=0.0, value=0.0, step=0.5, format="%.2f")
    noise = c3.number_input("noise std", min_value=0.0, value=0.0, step=0.001, format="%.4f")
    seed = c4.number_input("seed", min_value=0, value=42, step=1)

    if not st.button("評価を実行", key="run_evaluate"):
        return

    if not faults:
        st.info("故障させる素子を 1 つ以上選択してください。")
        return

    try:
        with st.spinner(f"{int(trials)} 試行を実行中..."):
            res = service.run_evaluate(
                circuit, faults, k,
                trials=int(trials), tol=float(tol), noise=float(noise),
                seed=int(seed), frequencies=frequencies,
            )
    except Exception as e:  # noqa: BLE001 - surface any failure in the UI
        st.error(f"エラー: {e}")
        return

    m1, m2 = st.columns(2)
    m1.metric("Top-1 accuracy", f"{res['top1_accuracy']*100:.1f}%")
    m2.metric("Top-3 accuracy", f"{res['top3_accuracy']*100:.1f}%")

    m3, m4 = st.columns(2)
    m3.metric("Ambiguous rate", f"{res['ambiguous_rate']*100:.1f}%")
    m4.metric("Ill-conditioned rate", f"{res['ill_conditioned_rate']*100:.1f}%")

    st.markdown(
        f"**Correct nodes**: {res['correct_nodes']}  \n"
        f"**Trials**: {res['trials']} / **tol**: {res['tol_percent']}% / "
        f"**noise**: {res['noise_std']} / **freq**: {res['frequencies'] or 'DC'}"
    )
