from pathlib import Path
from typing import List
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from src.generation.prompts import FINANCE_RAG_PROMPT, format_docs
from src.generation.llm import get_llm


def build_rag_chain(retriever):
    """Full RAGchain using LCEL"""
    llm = get_llm()

    rag_chain = (
        {
            "context": RunnableLambda(
                lambda x: format_docs(retriever.invoke(x["question"]))
            ),
            "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
        }
        | FINANCE_RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain


def build_rag_chain_with_sources(retriever):
    """
    Extended chain that returns both the answer AND the source documents.
    """
    llm = get_llm()

    def retrieve_and_format(inputs):
        question = inputs["question"]
        docs = retriever.invoke(question)
        formatted = format_docs(docs)
        return {"context": formatted, "question": question, "source_docs": docs}

    answer_chain = RunnableLambda(retrieve_and_format) | {
        "answer": (
            RunnableLambda(
                lambda x: {
                    "context": x["context"],
                    "question": x["question"],
                }
            )
            | FINANCE_RAG_PROMPT
            | llm
            | StrOutputParser()
        ),
        "source_docs": RunnableLambda(lambda x: x["source_docs"]),
    }
    return answer_chain

def ask(chain, question: str) -> str:
    """
    Ask a question and get an answer.
    """
    print(f"\nQuestion: {question}")
    print("Thinking...\n")

    answer = chain.invoke({"question": question})

    print(f"Answer: {answer}")
    return answer