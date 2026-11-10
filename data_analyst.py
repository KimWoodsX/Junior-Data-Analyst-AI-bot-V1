import os
import re
import warnings

import numpy as np
import pandas as pd
from langchain_core.tools import tool

print("⚡ Initializing Data Analyst Backend...")

gate_open = (not human_approval) or st.session_state.approved
can_chat = has_key and gate_open and st.session_state.payload is not None

# Render past chat messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("timing"):
            st.caption(m["timing"])
        if m.get("extra"):
            with st.expander("Extra facts sent with this question"):
                st.code(m["extra"])

# Human approval prompt if needed
if human_approval and not st.session_state.approved and st.session_state.payload:
    with st.expander("🔒 Review what will be sent to Groq (summary statistics only)", expanded=True):
        st.code(st.session_state.payload)
        if st.button("✅ Approve and start chatting"):
            st.session_state.approved = True
            st.rerun()

# Suggestions
if len(st.session_state.messages) <= 1 and can_chat:
    st.caption("Try one of these:")
    cols = st.columns(len(SUGGESTIONS))
    for col, s in zip(cols, SUGGESTIONS):
        if col.button(s, key=f"sug_{s}"):
            st.session_state.pending = s
            st.rerun()

typed = st.chat_input("Ask anything about your data…", disabled=not can_chat)
question = typed or st.session_state.pop("pending", None)

if question and can_chat:
    extra = question_facts(df, question)
    history = history_messages()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            t0, first = time.time(), []

            def timed(gen):
                for chunk in gen:
                    if not first:
                        first.append(time.time() - t0)
                    yield chunk

            with st.spinner("Thinking…"):
                streamed = st.write_stream(timed(stream_answer(
                    question, history, st.session_state.payload, extra, st.session_state.get("context", ""))))
            answer = re.sub(r"<think>.*?</think>", "", str(streamed), flags=re.DOTALL).strip()
            timing = f"⏱ first word {first[0] if first else 0:.1f}s · total {time.time() - t0:.1f}s"
            st.caption(timing)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "extra": extra, "timing": timing})
            st.rerun()
        except Exception as e:
            msg = f"⚠️ I couldn't get an answer: {e}"
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "local": True})

st.markdown("---")

# ------------------------------------------------------------------ 2. GRAPHS (BELOW THE CHAT)
st.subheader("📈 Visualizations & Charts")

nums, cats = split_columns(df)
fk = (st.session_state.file_sig or "none")[:8]

# Controls row
with st.expander("🛠️ Chart Controls & Dimensions", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        x_axis = st.selectbox("X-Axis / Timeline", df.columns.tolist(), index=guess_time_col(df), key=f"x_{fk}")
        theme = st.selectbox("Theme", THEMES, key=f"theme_{fk}")
    with c2:
        ys = st.multiselect("Numeric features (Y)", nums, default=nums[:2] if len(nums) >= 2 else nums, key=f"ys_{fk}")
        cat = st.selectbox("Category dimension", cats or df.columns.tolist(), key=f"cat_{fk}")
    with c3:
        color_by = st.selectbox("Color by", ["None"] + cats, key=f"color_{fk}")

# Render the Gallery of Graphs directly on screen below
gallery = [
    "Line Trend", "Correlation Heatmap",
    "Histogram", "Box (Outliers)",
    "Scatter (Relationship)", "Missing Values"
]

for i in range(0, len(gallery), 2):
    ga, gb = st.columns(2)
    for col, kind in zip((ga, gb), gallery[i:i + 2]):
        with col:
            render_chart(kind, df, x_axis, ys, cat, color_by, theme, f"main_gal_{kind}_{fk}")

st.markdown("---")

# ------------------------------------------------------------------ 3. DATA PREVIEW (BOTTOM)
with st.expander("📊 Raw Data Preview (First 1,000 Rows)", expanded=False):
    st.dataframe(df.head(1000), use_container_width=True)
    if st.session_state.backend:
        st.markdown("### Profile Summary")
        st.markdown(st.session_state.backend.get("narrative_text", ""))
