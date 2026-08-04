"""
YouTube AI Agent - Video Summarizer and Q&A Assistant
Entry point for the Streamlit application.

Run with:  streamlit run app.py
"""

import os
import streamlit as st

from utils.transcript import fetch_transcript, extract_video_id, chunk_transcript, TranscriptError
from utils.vectorstore import build_vectorstore, answer_question
from utils.ai_features import (
    generate_concise_summary,
    generate_detailed_summary,
    generate_key_points,
    generate_quiz,
    generate_interview_questions,
)
from utils.pdf_export import build_pdf
from utils.chat_history import load_history, append_message, clear_history


# --------------------------------------------------------------------------
# Page config & theme
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="YouTube AI Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
.stApp { background-color: #0e1117; color: #f0f0f0; }
.stTextInput input, .stTextArea textarea { background-color: #1c1f26; color: #f0f0f0; }
div.stButton > button { border-radius: 8px; }
</style>
"""

LIGHT_CSS = """
<style>
div.stButton > button { border-radius: 8px; }
</style>
"""


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
def init_state():
    defaults = {
        "transcript": None,
        "video_id": None,
        "video_url": "",
        "vectorstore": None,
        "concise_summary": None,
        "detailed_summary": None,
        "key_points": None,
        "quiz": None,
        "interview_questions": None,
        "dark_mode": True,
        "chat_messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()
st.markdown(DARK_CSS if st.session_state.dark_mode else LIGHT_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Sidebar: settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help="Your key is used only for this session and is not stored.",
    )

    st.session_state.dark_mode = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)

    st.markdown("---")
    st.caption(
        "Built with Streamlit, LangChain, OpenAI API, YouTube Transcript API, "
        "and FAISS (RAG)."
    )


# --------------------------------------------------------------------------
# Header + URL input
# --------------------------------------------------------------------------
st.title("🎬 YouTube AI Agent")
st.caption("Video Summarizer & Q&A Assistant — powered by LangChain + RAG")

col_url, col_fetch = st.columns([4, 1])
with col_url:
    video_url = st.text_input(
        "YouTube Video URL", placeholder="https://www.youtube.com/watch?v=..."
    )
with col_fetch:
    st.write("")
    st.write("")
    fetch_clicked = st.button("📥 Load Video", use_container_width=True)

if fetch_clicked:
    if not api_key_input:
        st.error("Please enter your OpenAI API key in the sidebar first.")
    elif not video_url:
        st.error("Please paste a YouTube video URL.")
    else:
        with st.spinner("Fetching transcript..."):
            try:
                transcript = fetch_transcript(video_url)
                video_id = extract_video_id(video_url)
                st.session_state.transcript = transcript
                st.session_state.video_id = video_id
                st.session_state.video_url = video_url
                # Reset previously generated content for the new video
                for key in [
                    "concise_summary", "detailed_summary", "key_points",
                    "quiz", "interview_questions", "vectorstore",
                ]:
                    st.session_state[key] = None
                st.session_state.chat_messages = load_history(video_id)
                st.success(f"Transcript loaded ({len(transcript.split())} words).")
            except TranscriptError as e:
                st.error(str(e))


# --------------------------------------------------------------------------
# Main feature tabs (only shown once a transcript is loaded)
# --------------------------------------------------------------------------
if st.session_state.transcript:
    tabs = st.tabs(
        ["📝 Summary", "🔑 Key Points", "🧩 Quiz", "💬 Ask Questions", "🎯 Interview Prep"]
    )

    # ---- Summary tab ----
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Concise Summary"):
                with st.spinner("Generating concise summary..."):
                    st.session_state.concise_summary = generate_concise_summary(
                        st.session_state.transcript, api_key_input
                    )
        with col2:
            if st.button("Generate Detailed Summary"):
                with st.spinner("Generating detailed summary..."):
                    st.session_state.detailed_summary = generate_detailed_summary(
                        st.session_state.transcript, api_key_input
                    )

        if st.session_state.concise_summary:
            st.subheader("Concise Summary")
            st.write(st.session_state.concise_summary)

        if st.session_state.detailed_summary:
            st.subheader("Detailed Summary")
            st.write(st.session_state.detailed_summary)

        if st.session_state.concise_summary or st.session_state.detailed_summary:
            pdf_bytes = build_pdf(
                title="YouTube AI Agent - Summary Report",
                video_url=st.session_state.video_url,
                sections={
                    "Concise Summary": st.session_state.concise_summary,
                    "Detailed Summary": st.session_state.detailed_summary,
                    "Key Points": st.session_state.key_points,
                },
            )
            st.download_button(
                "⬇️ Download Summary as PDF",
                data=pdf_bytes,
                file_name="youtube_summary.pdf",
                mime="application/pdf",
            )

    # ---- Key Points tab ----
    with tabs[1]:
        if st.button("Extract Key Points"):
            with st.spinner("Extracting key points..."):
                st.session_state.key_points = generate_key_points(
                    st.session_state.transcript, api_key_input
                )
        if st.session_state.key_points:
            st.write(st.session_state.key_points)

    # ---- Quiz tab ----
    with tabs[2]:
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz..."):
                st.session_state.quiz = generate_quiz(
                    st.session_state.transcript, api_key_input
                )
        if st.session_state.quiz:
            st.write(st.session_state.quiz)

    # ---- Ask Questions (RAG) tab ----
    with tabs[3]:
        st.caption(
            "Answers are generated using Retrieval-Augmented Generation (RAG) "
            "over the video transcript via a FAISS vector index."
        )

        if st.session_state.vectorstore is None:
            if st.button("🔎 Build Search Index for this Video"):
                with st.spinner("Chunking transcript and building FAISS index..."):
                    chunks = chunk_transcript(st.session_state.transcript)
                    st.session_state.vectorstore = build_vectorstore(chunks, api_key_input)
                st.success("Index ready — you can ask questions now.")
        else:
            st.success("Search index ready.")

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("Ask something about this video...")
        if question:
            if st.session_state.vectorstore is None:
                st.warning("Build the search index first (button above).")
            else:
                st.session_state.chat_messages.append({"role": "user", "content": question})
                append_message(st.session_state.video_id, "user", question)
                with st.spinner("Thinking..."):
                    answer = answer_question(
                        st.session_state.vectorstore, question, api_key_input
                    )
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                append_message(st.session_state.video_id, "assistant", answer)
                st.rerun()

        if st.session_state.chat_messages:
            if st.button("🗑️ Clear Chat History"):
                clear_history(st.session_state.video_id)
                st.session_state.chat_messages = []
                st.rerun()

    # ---- Interview Prep tab ----
    with tabs[4]:
        if st.button("Generate Interview Questions"):
            with st.spinner("Generating interview questions..."):
                st.session_state.interview_questions = generate_interview_questions(
                    st.session_state.transcript, api_key_input
                )
        if st.session_state.interview_questions:
            st.write(st.session_state.interview_questions)

else:
    st.info("👆 Paste a YouTube URL and click **Load Video** to get started.")
