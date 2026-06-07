"""Testability tab: k-node testability check."""
import streamlit as st

from fault import service


def render_testability(circuit, k: int):
    st.subheader("k-node テスタビリティ判定")
    st.caption(f"共通パラメータ k = {k}（サイドバーで変更）")

    if st.button("判定を実行", key="run_testability"):
        try:
            res = service.run_testability(circuit, k)
        except Exception as e:  # noqa: BLE001 - surface any failure in the UI
            st.error(f"エラー: {e}")
            return

        if res["testable"]:
            st.success(f"Testable: True （k={k} まで診断可能）")
        else:
            st.warning(f"Testable: False （k={k} の診断には独立パスが不足）")

        st.markdown("**内部ノードの接続度（accessible/ground への独立パス数）**")
        rows = [
            {"node": node, "connectivity": conn, "needs (k+1)": k + 1,
             "ok": conn >= k + 1}
            for node, conn in res["connectivities"].items()
        ]
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("内部（アクセス不可）ノードがありません。")
