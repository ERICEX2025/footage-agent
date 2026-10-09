import os
import re

import streamlit as st
import weave
from dotenv import load_dotenv

from agent import FootageAgent, _wandb_project

load_dotenv()
st.set_page_config(page_title="Food Court Watch", layout="wide")


@st.cache_resource
def get_agent():
    if _wandb_project():
        weave.init(_wandb_project())
    return FootageAgent()


agent = get_agent()
st.title("Food Court Watch")
st.caption("Loss-prevention agent for small food businesses · Cosmos-Reason + YOLO11 on VAST · reasoning by W&B Inference")

if "history" not in st.session_state:
    st.session_state.history = []

for m in st.session_state.history:
    st.chat_message(m["role"]).markdown(m["content"])

question = st.chat_input("e.g. Did anyone leave with food without paying today?")
if question:
    st.chat_message("user").markdown(question)
    agent.evidence = {}
    with st.chat_message("assistant"):
        with st.status("Investigating...", expanded=True) as status:
            answer = agent.run(
                question,
                history=st.session_state.history[-6:],
                on_step=lambda name, args: st.write(f"`{name}` {args}"),
            )
            status.update(label="Done", state="complete", expanded=False)
        st.markdown(answer)

    st.session_state.history += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    # Show the clips the answer cites (fall back to top search hits).
    cited = [s for s in agent.evidence.values() if s["filename"] and s["filename"] in answer]
    clips = (cited or sorted(agent.evidence.values(), key=lambda s: -(s["score"] or 0)))[:6]
    if clips:
        st.subheader("Evidence")
        cols = st.columns(3)
        for i, s in enumerate(clips):
            with cols[i % 3]:
                st.video(agent.vss.stream_url(s["source"]))
                desc = re.sub(r"\s+", " ", s["description"])[:200]
                st.caption(f"**{s['filename']}** @ {s['start']}-{s['end']}s · score {s['score']}\n\n{desc}")
                if s["objects"]:
                    st.caption(f"YOLO: {s['objects']}")
