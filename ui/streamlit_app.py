"""Small Streamlit client for the local AntiBros API."""

from __future__ import annotations

import os

import requests
import streamlit as st

API = os.getenv("ANTIBROS_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AntiBros", page_icon="🏋️", layout="wide")
st.title("AntiBros")
st.caption("Protein & creatine answers grounded in open-access research")

question = st.text_area("Your question / Câu hỏi", height=100, max_chars=4000)
if st.button("Ask / Hỏi", type="primary", disabled=len(question.strip()) < 3):
    try:
        response = requests.post(f"{API}/ask", json={"question": question}, timeout=180)
        response.raise_for_status()
        st.session_state["last_answer"] = response.json()
    except requests.RequestException as exc:
        st.error(f"API is unavailable: {exc}")

result = st.session_state.get("last_answer")
if result:
    status = result["status"]
    badge = {
        "answered": "Answered",
        "abstained": "Insufficient evidence",
        "safety_redirect": "Safety",
    }.get(status, status)
    st.markdown(f"**{badge} · {result['confidence']} confidence**")
    st.markdown(result["answer"])

    with st.expander("Sources / Nguồn", expanded=True):
        for citation in result["citations"]:
            st.markdown(
                f"**[{citation['id']}] {citation['title']}** ({citation['year']})  \n"
                f"{citation['journal']} · Evidence: **{citation['evidence_label']}**  \n"
                f"[Open paper]({citation['url']}) · `{citation['chunk_id']}`"
            )
            st.caption(citation["snippet"])

    with st.expander("Debug"):
        st.json(result["debug"])
        st.caption(f"Trace `{result['trace_id']}` · {result['latency_ms']} ms")

    rating = st.feedback("stars", key=f"rating_{result['trace_id']}")
    if rating is not None and st.button("Send feedback"):
        try:
            feedback = requests.post(
                f"{API}/feedback",
                json={"trace_id": result["trace_id"], "rating": rating + 1},
                timeout=10,
            )
            feedback.raise_for_status()
            st.success("Feedback received")
        except requests.RequestException as exc:
            st.error(f"Feedback could not be sent: {exc}")
