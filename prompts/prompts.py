"""
Prompt templates for the YouTube AI Agent.

Each function returns a LangChain PromptTemplate. Keeping prompts in one
module makes them easy to tune independently of the app logic.
"""

from langchain_classic.prompts import PromptTemplate


def concise_summary_prompt() -> PromptTemplate:
    template = """You are an expert video summarizer.
Using ONLY the transcript below, write a concise summary between 100 and 200 words.
Focus on the main ideas and overall purpose of the video. Do not add information
that is not in the transcript.

Transcript:
{transcript}

Concise Summary (100-200 words):"""
    return PromptTemplate(input_variables=["transcript"], template=template)


def detailed_summary_prompt() -> PromptTemplate:
    template = """You are an expert video summarizer.
Using ONLY the transcript below, write a detailed, well-structured summary that
captures the flow of the video: introduction, main sections/arguments, examples
given, and conclusion. Use short paragraphs or headings where useful.

Transcript:
{transcript}

Detailed Summary:"""
    return PromptTemplate(input_variables=["transcript"], template=template)


def key_points_prompt() -> PromptTemplate:
    template = """You are an expert note-taker.
Read the transcript below and extract exactly 10 key learning points.
Return them as a numbered list (1-10). Each point should be a single,
self-contained sentence capturing one important idea. Do not repeat points.

Transcript:
{transcript}

10 Key Points:"""
    return PromptTemplate(input_variables=["transcript"], template=template)


def quiz_generation_prompt() -> PromptTemplate:
    template = """You are an assessment designer.
Based ONLY on the transcript below, create exactly 10 multiple-choice questions (MCQs)
that test understanding of the video's content.

Format each question EXACTLY like this:

Q1. <question text>
A) <option>
B) <option>
C) <option>
D) <option>
Answer: <correct option letter>

Make sure only one option is correct, options are plausible, and questions cover
different parts of the transcript (not all from the same section).

Transcript:
{transcript}

Quiz:"""
    return PromptTemplate(input_variables=["transcript"], template=template)


def interview_questions_prompt() -> PromptTemplate:
    template = """You are a technical/educational interviewer.
Based on the educational content in the transcript below, generate 8-10 interview-style
questions that a recruiter or teacher could ask to test deep understanding of the topic
discussed. Mix conceptual, applied, and "explain in your own words" style questions.
Return as a numbered list.

Transcript:
{transcript}

Interview Questions:"""
    return PromptTemplate(input_variables=["transcript"], template=template)


def rag_qa_prompt() -> PromptTemplate:
    """Used with retrieved context chunks (RAG), not the full transcript."""
    template = """You are a helpful assistant answering questions about a YouTube video.
Use ONLY the context below, which was retrieved from the video's transcript, to answer
the question. If the answer is not contained in the context, say you don't have enough
information from the video to answer that.

Context:
{context}

Question:
{question}

Answer:"""
    return PromptTemplate(input_variables=["context", "question"], template=template)
