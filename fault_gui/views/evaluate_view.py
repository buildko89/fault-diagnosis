"""Evaluate tab: Monte Carlo accuracy under tolerance + noise, with optional sweep."""
import numpy as np
import streamlit as st

from .. import cached
from ..plots import accuracy_sweep_figure
from ..widgets import fault_input


def _freq_key(frequencies):
    return tuple(frequencies) if frequencies else None


def render_evaluate(circuit, circuit_key: str, k: int, frequencies):
    st.subheader("モンテカルロ評価")
    st.caption(f"共通パラメータ k = {k} / frequencies = {frequencies or 'DC'}（サイドバーで変更）")

    faults = fault_input(circuit.config.elements, key_prefix="eval")

    c1, c2, c3, c4 = st.columns(4)
    trials = c1.number_input("trials", min_value=1, max_value=10000, value=100, step=10)
    tol = c2.number_input("tolerance [%]", min_value=0.0, value=0.0, step=0.5, format="%.2f")
    noise = c3.number_input("noise std", min_value=0.0, value=0.0, step=0.001, format="%.4f")
    seed = c4.number_input("seed", min_value=0, value=42, step=1)

    faults_items = tuple(sorted(faults.items()))
    freq_key = _freq_key(frequencies)

    tab_single, tab_sweep = st.tabs(["単発評価", "掃引プロット"])

    # ---- single evaluation ------------------------------------------------
    with tab_single:
        if st.button("評価を実行", key="run_evaluate"):
            if not faults:
                st.info("故障させる素子を 1 つ以上選択してください。")
            else:
                try:
                    with st.spinner(f"{int(trials)} 試行を実行中..."):
                        res = cached.evaluate(circuit_key, faults_items, k,
                                              int(trials), float(tol), float(noise),
                                              int(seed), freq_key)
                except Exception as e:  # noqa: BLE001
                    st.error(f"エラー: {e}")
                else:
                    _show_metrics(res)

    # ---- parameter sweep --------------------------------------------------
    with tab_sweep:
        param = st.selectbox("掃引するパラメータ", ["tolerance [%]", "noise std"], key="sweep_param")
        s1, s2, s3 = st.columns(3)
        start = s1.number_input("start", min_value=0.0, value=0.0, step=0.5, format="%.4f", key="sw_start")
        stop = s2.number_input("stop", min_value=0.0, value=5.0, step=0.5, format="%.4f", key="sw_stop")
        steps = s3.number_input("points", min_value=2, max_value=30, value=6, step=1, key="sw_steps")

        if st.button("掃引を実行", key="run_sweep"):
            if not faults:
                st.info("故障させる素子を 1 つ以上選択してください。")
            elif stop <= start:
                st.warning("stop は start より大きくしてください。")
            else:
                xs = np.linspace(float(start), float(stop), int(steps))
                ys = []
                progress = st.progress(0.0, text="掃引中...")
                try:
                    for i, x in enumerate(xs):
                        t = float(x) if param == "tolerance [%]" else float(tol)
                        n = float(x) if param == "noise std" else float(noise)
                        res = cached.evaluate(circuit_key, faults_items, k,
                                              int(trials), t, n, int(seed), freq_key)
                        ys.append(res["top1_accuracy"] * 100.0)
                        progress.progress((i + 1) / len(xs), text=f"{i+1}/{len(xs)}")
                except Exception as e:  # noqa: BLE001
                    progress.empty()
                    st.error(f"エラー: {e}")
                else:
                    progress.empty()
                    st.pyplot(accuracy_sweep_figure(xs, ys, xlabel=param))
                    st.dataframe(
                        [{"param": float(x), "top1_accuracy [%]": round(y, 1)} for x, y in zip(xs, ys)],
                        width="stretch", hide_index=True,
                    )


def _show_metrics(res):
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
