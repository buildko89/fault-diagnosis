"""Circuit tab: topology figure + summary table."""
import streamlit as st

from fault import service
from ..plots import topology_figure


def render_circuit(circuit):
    st.subheader("回路トポロジ")
    summary = service.circuit_summary(circuit)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.pyplot(topology_figure(circuit))
    with col2:
        st.markdown(
            f"**Name**: `{summary['name']}`  \n"
            f"**Reference**: `{summary['reference']}`  \n"
            f"**Nodes**: {summary['nodes']}  \n"
            f"**Accessible (ADC)**: {summary['accessible']}  \n"
            f"**Inaccessible**: {summary['inaccessible']}  \n"
            f"**Frequencies [Hz]**: {summary['frequencies'] or 'DC'}"
        )

    st.markdown("**素子一覧**")
    st.dataframe(summary["elements"], width="stretch", hide_index=True)
