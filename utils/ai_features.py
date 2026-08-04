"""
Wraps the LangChain LLM calls for each AI feature: summaries, key points,
quiz generation, and interview questions. Each function takes the raw
transcript text and returns a plain string result.
"""

from langchain_community.chat_models.openai import ChatOpenAI
from langchain_classic.chains.llm import LLMChain

from prompts.prompts import (
    concise_summary_prompt,
    detailed_summary_prompt,
    key_points_prompt,
    quiz_generation_prompt,
    interview_questions_prompt,
)

MAX_TRANSCRIPT_CHARS = 15000  # keep prompts within a safe token budget


def _get_llm(openai_api_key: str, model_name: str = "gpt-4o-mini", temperature: float = 0.4):
    return ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=openai_api_key)


def _truncate(transcript: str) -> str:
    if len(transcript) <= MAX_TRANSCRIPT_CHARS:
        return transcript
    return transcript[:MAX_TRANSCRIPT_CHARS] + " ...[transcript truncated for length]"


def generate_concise_summary(transcript: str, openai_api_key: str) -> str:
    llm = _get_llm(openai_api_key, temperature=0.3)
    chain = LLMChain(llm=llm, prompt=concise_summary_prompt())
    return chain.run(transcript=_truncate(transcript)).strip()


def generate_detailed_summary(transcript: str, openai_api_key: str) -> str:
    llm = _get_llm(openai_api_key, temperature=0.3)
    chain = LLMChain(llm=llm, prompt=detailed_summary_prompt())
    return chain.run(transcript=_truncate(transcript)).strip()


def generate_key_points(transcript: str, openai_api_key: str) -> str:
    llm = _get_llm(openai_api_key, temperature=0.3)
    chain = LLMChain(llm=llm, prompt=key_points_prompt())
    return chain.run(transcript=_truncate(transcript)).strip()


def generate_quiz(transcript: str, openai_api_key: str) -> str:
    llm = _get_llm(openai_api_key, temperature=0.5)
    chain = LLMChain(llm=llm, prompt=quiz_generation_prompt())
    return chain.run(transcript=_truncate(transcript)).strip()


def generate_interview_questions(transcript: str, openai_api_key: str) -> str:
    llm = _get_llm(openai_api_key, temperature=0.5)
    chain = LLMChain(llm=llm, prompt=interview_questions_prompt())
    return chain.run(transcript=_truncate(transcript)).strip()
