"""Diagnose tab: simulate a fault and locate it, with figures."""
import streamlit as st

from fault import service
from ..plots import topology_figure, delta_v_figure
from ..widgets import fault_input


_STATUS_RENDER = {
    "unique": ("success", "unique（一意に特定）"),
    "ambiguous": ("warning", "ambiguous（候補が拮抗）"),
    "ill_conditioned": ("warning", "ill_conditioned（悪条件）"),
    "no_candidates": ("error", "no_candidates（候補なし）"),
}


def _fmt(x, spec="%.6e"):
    try:
        if x != x or x in (float("inf"), float("-inf")):  # NaN/inf guard
            return str(x)
        return spec % x
    except (TypeError, ValueError):
        return str(x)


def render_diagnose(circuit, k: int, frequencies):
    st.subheader("故障診断")
    st.caption(f"共通パラメータ k = {k} / frequencies = {frequencies or 'DC'}（サイドバーで変更）")

    faults = fault_input(circuit.config.elements, key_prefix="diag")

    col1, col2, col3 = st.columns(3)
    method = col1.selectbox("method", ["auto", "exhaustive", "omp"], index=0)
    top_n = col2.number_input("top-n", min_value=1, max_value=50, value=5, step=1)
    reconstruct = col3.checkbox("枝を再構築する（R/C/L 種別）", value=False)

    if not st.button("診断を実行", key="run_diagnose"):
        return

    if not faults:
        st.info("故障させる素子を 1 つ以上選択してください。")
        return

    try:
        result = service.run_diagnose(
            circuit, faults, k, top_n=int(top_n), method=method,
            frequencies=frequencies, reconstruct=reconstruct,
        )
    except Exception as e:  # noqa: BLE001 - surface any failure in the UI
        st.error(f"エラー: {e}")
        return

    kind, label = _STATUS_RENDER.get(result["status"], ("info", result["status"]))
    getattr(st, kind)(f"Status: {label}")

    best = result.get("best")
    faulty_nodes = best["support"] if best else []

    if best:
        m1, m2, m3 = st.columns(3)
        m1.metric("Faulty nodes", str(best["support"]))
        m2.metric("Residual", _fmt(best["residual_norm"]))
        m3.metric("Cond. number", _fmt(best["condition_number"], "%.2f"))

    st.markdown("**Top candidates**")
    cand_rows = [
        {
            "rank": i + 1,
            "support": str(c["support"]),
            "residual_norm": _fmt(c["residual_norm"]),
            "relative_residual": _fmt(c["relative_residual"]),
            "condition_number": _fmt(c["condition_number"], "%.2f"),
        }
        for i, c in enumerate(result["candidates"])
    ]
    st.dataframe(cand_rows, width="stretch", hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(topology_figure(circuit, faulty_nodes))
    with col_b:
        st.pyplot(delta_v_figure(circuit, result["delta_v_m"]))

    if reconstruct and result.get("branch"):
        st.markdown("**枝アドミタンス再構築**")
        st.dataframe(_branch_rows(result["branch"]),
                     width="stretch", hide_index=True)


def _branch_rows(branch):
    rows = []
    for name, info in branch.items():
        if "delta_g" in info:  # DC
            rows.append({
                "branch": name,
                "nominal_g": info.get("nominal_g"),
                "delta_g": _fmt(info["delta_g"]),
                "estimated_g": _fmt(info.get("estimated_g")),
            })
        else:  # AC
            rows.append({
                "branch": name,
                "type": info.get("type"),
                "classification": info.get("classification", "-"),
                "delta_value": _fmt(info.get("delta_value")) if "delta_value" in info else "-",
            })
    return rows
