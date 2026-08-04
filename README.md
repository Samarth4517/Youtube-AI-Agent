# 🎬 YouTube AI Agent — Video Summarizer & Q&A Assistant

A full-stack Generative AI application that takes any YouTube video URL and lets you:

- Generate a **concise (100–200 word)** or **detailed** summary
- Extract **10 key learning points**
- Auto-generate a **10-question MCQ quiz** (with answers)
- **Ask free-form questions** about the video using Retrieval-Augmented Generation (RAG)
- Generate **interview-style questions** from educational content
- **Download** the generated summary as a PDF
- **Save chat history** per video

Built with **Python, Streamlit, LangChain, OpenAI API, YouTube Transcript API, and FAISS**.

---

## 🧱 Architecture

```
User pastes YouTube URL
        │
        ▼
YouTube Transcript API ──► raw transcript text
        │
        ├──► LangChain + OpenAI (gpt-4o-mini) ──► Summary / Key Points / Quiz / Interview Qs
        │
        └──► Text splitter ──► OpenAI Embeddings ──► FAISS vector index
                                                          │
                                    User question ──► Retriever (top-k chunks)
                                                          │
                                                          ▼
                                              LangChain RetrievalQA (RAG) ──► Answer
```

## 📁 Folder Structure

```
youtube-ai-agent/
├── app.py                  # Streamlit UI and orchestration
├── utils/
│   ├── transcript.py       # Video ID extraction + transcript fetching/chunking
│   ├── vectorstore.py      # FAISS index building + RAG question answering
│   ├── ai_features.py      # LangChain chains for summary/key points/quiz/interview Qs
│   ├── pdf_export.py       # PDF report generation (reportlab)
│   └── chat_history.py     # JSON-based per-video chat history persistence
├── prompts/
│   └── prompts.py          # All prompt templates (prompt engineering lives here)
├── data/
│   └── chat_history.json   # Auto-created at runtime
├── requirements.txt
└── README.md
```

## 🚀 Installation

1. **Clone / copy the project folder**, then create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your OpenAI API key** (either method works):
   - Enter it directly in the app's sidebar at runtime, **or**
   - Create a `.env` file in the project root:
     ```
     OPENAI_API_KEY=sk-your-key-here
     ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

5. Open the URL Streamlit prints (usually `http://localhost:8501`), paste a YouTube video URL, click **Load Video**, and explore the tabs.

## ⚠️ Notes & Limitations

- Works only on videos that have **captions/transcripts available** (auto-generated or manual). Videos with transcripts disabled will show a friendly error.
- Very long videos are truncated to a safe character budget before summarization to control token usage/cost — the RAG Q&A tab still searches the **full** transcript via chunking, so long-video Q&A stays accurate even when the summary is truncated.
- Uses `gpt-4o-mini` by default for a good cost/quality balance; swap the `model_name` in `utils/ai_features.py` / `utils/vectorstore.py` for a different model.
- Chat history is stored locally in `data/chat_history.json` — fine for a personal/demo project, not intended as production-grade storage.

## 🧠 Resume Bullet

> Developed an AI-powered YouTube Agent that extracts video transcripts, generates intelligent summaries, creates quizzes, and answers user questions using Generative AI. Implemented LangChain, OpenAI API, Prompt Engineering, and RAG architecture with FAISS vector search to enable natural language interaction with YouTube content.
