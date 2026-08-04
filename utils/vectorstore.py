"""
Builds a FAISS vector store from transcript chunks and exposes a
Retrieval-Augmented Generation (RAG) question-answering helper.
"""

from typing import List

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from prompts.prompts import rag_qa_prompt


def build_vectorstore(chunks: List[str], openai_api_key: str) -> FAISS:
    """
    Embed transcript chunks with OpenAI embeddings and build an in-memory
    FAISS index. This index lives for the duration of the Streamlit session
    (stored in st.session_state by the caller).
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=openai_api_key
    )
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore


def answer_question(
    vectorstore: FAISS,
    question: str,
    openai_api_key: str,
    model_name: str = "gpt-4o-mini",
    k: int = 4,
) -> str:
    """
    Run a RAG query: retrieve the top-k relevant transcript chunks for the
    question, then answer using only that retrieved context.
    """
    llm = ChatOpenAI(
        model=model_name, temperature=0.2, openai_api_key=openai_api_key
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": rag_qa_prompt().partial()},
        return_source_documents=False,
    )

    # RetrievalQA expects {"query": ...} and its prompt uses {context}/{question}
    # internally when built via from_chain_type with a custom prompt that uses
    # "context" and "question" keys — LangChain maps "query" -> "question" is NOT
    # automatic for custom prompts, so we call the chain with the right key.
    result = qa_chain.invoke({"query": question})
    return result["result"]
