# 🎬 YouTube AI Agent — Video Summarizer & Q&A Assistant

A demo app that turns any YouTube video into an AI-powered study assistant.

**Features:**
- Generate a **concise summary** or a **detailed summary** of the video transcript.
- Extract **10 key learning points**.
- Create a **10-question multiple-choice quiz** with answers.
- Generate **interview-style questions**.
- Ask **free-form questions** about the video using Retrieval-Augmented Generation (RAG).
- **Download summary content as a PDF**.
- **Persist chat history per video** locally.

Built with **Python, Streamlit, LangChain, OpenAI API, YouTube Transcript API, and FAISS**.

---

## 🚀 What it does

1. User pastes a YouTube URL.
2. The app fetches the transcript using the YouTube Transcript API.
3. It uses LangChain + OpenAI to generate summaries, quizzes, key points, and interview questions.
4. It builds a FAISS vector index over transcript chunks for RAG-powered question answering.
5. Users can ask questions and receive answers based only on retrieved transcript context.

---

## 📁 Project structure

```
youtube-ai-agent/
├── app.py                  # Streamlit UI and feature orchestration
├── requirements.txt       # Python dependency pins
├── README.md              # Project documentation
├── utils/
│   ├── transcript.py      # Video ID extraction, transcript fetching, and chunking
│   ├── vectorstore.py     # FAISS index building and retrieval QA
│   ├── ai_features.py     # Summary, quiz, interview question generation
│   ├── pdf_export.py      # PDF generation using reportlab
│   └── chat_history.py    # Local history persistence per video
├── prompts/
│   └── prompts.py         # Prompt templates for all AI features
├── data/                  # Local app-generated data (e.g. chat history)
└── .gitignore
```

---

## 🛠️ Installation

```bash
cd youtube-ai-agent
python -m venv .venv
.venv\Scripts\activate  # Windows
# OR
source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

---

## 🔑 OpenAI setup

The app reads your OpenAI API key from one of these sources:

- Enter it in the Streamlit sidebar at runtime.
- Set an environment variable:
  ```bash
  set OPENAI_API_KEY=sk-your-key-here      # Windows PowerShell
  export OPENAI_API_KEY=sk-your-key-here   # macOS/Linux
  ```
- Or use an `.env` file with `python-dotenv` support:
  ```text
  OPENAI_API_KEY=sk-your-key-here
  ```

---

## ▶️ Run the app

```bash
streamlit run app.py
```

Then open the URL printed by Streamlit, usually `http://localhost:8501`.

---

## ⚠️ Notes

- The app requires a YouTube video with an available transcript.
- Summaries are truncated to avoid excessive token usage, but question answering still uses chunked transcript context.
- The default model is `gpt-4o-mini`; you can swap the model in `utils/ai_features.py` and `utils/vectorstore.py`.
- Chat history is stored locally in `data/chat_history.json`, so this is best for demo or personal use, not production.

---

## 📌 GitHub
https://github.com/Samarth4517/Youtube-AI-Agent

---

## 🧠 Summary
A YouTube content assistant that extracts transcripts, creates summaries and quizzes, and answers video questions using OpenAI, LangChain, and FAISS-powered retrieval.
