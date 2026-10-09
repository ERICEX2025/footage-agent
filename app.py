import json
import re

import streamlit as st
import streamlit.components.v1 as components
import weave
from dotenv import load_dotenv

from agent import REPORT_PROMPT, FootageAgent, _wandb_project
from voice import transcribe

load_dotenv()
st.set_page_config(page_title="SafeFloor", page_icon="🦺", layout="wide")


@st.cache_resource
def get_agent():
    try:  # tracing is nice-to-have; never block the app on it
        if _wandb_project():
            weave.init(_wandb_project())
    except Exception as e:
        print("weave.init failed:", e)
    return FootageAgent()


def speak(text):
    """Read the reply aloud with the browser's built-in speech synthesis."""
    components.html(f"""<script>
      const u = new SpeechSynthesisUtterance({json.dumps(text)});
      u.rate = 1.05;
      window.parent.speechSynthesis.cancel();
      window.parent.speechSynthesis.speak(u);
    </script>""", height=0)


agent = get_agent()
st.title("🦺 SafeFloor")
st.markdown("**An AI safety officer for your warehouse.** It watches the cameras you already have, "
            "catches near-misses before they become injuries, and answers questions by voice.")
st.caption("NVIDIA Cosmos-Reason + YOLO11 on VAST AI OS · NVIDIA Canary speech · W&B Inference agent")

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.last_audio = None

for m in st.session_state.history:
    st.chat_message(m["role"]).markdown(m["content"])

col_report, col_mic, col_hint = st.columns([1, 1, 2])
with col_report:
    report_clicked = st.button("📋 Today's safety report", type="primary", use_container_width=True)
with col_mic:
    audio = st.audio_input("Ask by voice")
with col_hint:
    st.markdown("Try: *“Any close calls between forklifts and people?”* · "
                "*“Is anything blocking the walkways?”* · *“Show me someone walking behind a forklift.”*")
typed = st.chat_input("...or type your question")

question = typed
if report_clicked:
    question = REPORT_PROMPT
if audio is not None and audio.getvalue() != st.session_state.last_audio:
    st.session_state.last_audio = audio.getvalue()
    with st.spinner("Listening (Canary-1B)..."):
        question = transcribe(audio.getvalue())

if question:
    st.chat_message("user").markdown("📋 Today's safety report" if question == REPORT_PROMPT else question)
    agent.evidence = {}
    with st.chat_message("assistant"):
        with st.status("Reviewing camera footage...", expanded=True) as status:
            answer = agent.run(
                question,
                max_steps=14 if question == REPORT_PROMPT else 8,
                history=st.session_state.history[-6:],
                on_step=lambda name, args: st.write(f"`{name}` {args}"),
            )
            status.update(label="Done", state="complete", expanded=False)
        st.markdown(answer)
        speak(answer.split("---")[0].strip())

    st.session_state.history += [
        {"role": "user", "content": "Generate today's safety report." if question == REPORT_PROMPT else question},
        {"role": "assistant", "content": answer},
    ]

    # Show the clips the answer cites (fall back to top search hits).
    cited = [s for s in agent.evidence.values() if s["filename"] and s["filename"] in answer]
    clips = (cited or sorted(agent.evidence.values(), key=lambda s: -(s["score"] or 0)))[:6]
    if clips:
        st.subheader("Evidence clips")
        cols = st.columns(3)
        for i, s in enumerate(clips):
            with cols[i % 3]:
                try:
                    st.video(agent.vss.clip_bytes(s["source"]))
                except Exception as e:
                    st.warning(f"Clip unavailable: {e}")
                desc = re.sub(r"\s+", " ", s["description"])[:200]
                where = " · ".join(x for x in (s.get("location"), s.get("camera")) if x)
                st.caption(f"**{s['filename']}** @ {s['start']}-{s['end']}s · {where}\n\n{desc}")
                if s["objects"]:
                    st.caption(f"YOLO: {s['objects']}")
