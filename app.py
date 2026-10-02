"""Minimal chat frontend. Run: streamlit run app.py"""
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from advisor.config import POLICY_DIR, get_llm  # noqa: E402
from advisor.graph import build_graph  # noqa: E402
from advisor.policy import PolicyStore  # noqa: E402

st.set_page_config(page_title="Weather-Advisory Bot", page_icon="🌦️")


@st.cache_resource
def get_app():
    store = PolicyStore(POLICY_DIR)
    return build_graph(get_llm(), store), store   # one shared graph + MemorySaver; one thread_id per browser session


graph, store = get_app()

if "thread_id" not in st.session_state:
    st.session_state.thread_id, st.session_state.history = str(uuid.uuid4()), []

with st.sidebar:
    st.header("Session")
    if st.button("New chat (clears memory)"):
        st.session_state.thread_id, st.session_state.history = str(uuid.uuid4()), []
        st.rerun()
    store.get()   # picks up edited policy files
    if store.last_error:
        st.error("Last policy edit was REJECTED; serving previous good policies:\n\n" + store.last_error)
    else:
        st.caption(f"{len(store.get().sops)} SOPs loaded (edits to policies/*.yaml apply on the next message)")

st.title("🌦️ Weather-Advisory Bot")
st.caption("Advice comes only from written SOPs. Ask e.g. “Is it safe to bike to work in Bhopal today?”")

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("debug"):
            with st.expander("Why this answer?"):
                st.json(m["debug"])

if prompt := st.chat_input("Ask about an outdoor activity..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Checking weather and policies..."):
            res = graph.invoke({"user_input": prompt}, config={"configurable": {"thread_id": st.session_state.thread_id}})
        st.markdown(res["reply"])
        debug = {"outcome": res["outcome"], "path": res["trace"], "sops": [s["id"] for s in res.get("selected", [])],
                 "facts_used": res.get("shown_facts"), "intent": res.get("intent"),
                 "guard_problems": res.get("problems"), "used_deterministic_fallback": res.get("used_fallback")}
        with st.expander("Why this answer?"):
            st.json(debug)
    st.session_state.history.append({"role": "assistant", "content": res["reply"], "debug": debug})
