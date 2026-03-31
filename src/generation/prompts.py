from langchain.prompts import PipelinePromptTemplate, PromptTemplate

FINANCE_RAGT_TEMPLATE="""You are a financail analu=yst assistant
Answer the question using ONLY the context provided below from SEC fillings.
Do not use any outside knowledge. If the answer is not present in the context, respond with: "I could not find this information in the provided filings.

Context:
{context}

Question: {question}

Instructions:
- Be precise with numbers, percentages, and dates
- Cite which company and filing year your answer comes from
- Keep your answer concise and to the factual

Answer:"""

FINANCE_RAG_PROMPT=PromptTemplate(template=FINANCE_RAGT_TEMPLATE, input_variables=["context", "question"],)


def format_docs(docs)->str:
    """
    Format retrieved documents into a single context string.

    source metadata (ticker, year, page) before each
    chunk so the LLM knows which document each piece of text came from.
    """

    formatted=[]

    for i, doc in enumerate(docs):
        ticker=doc.metadata.get("ticker", "UNKNOWN")
        year=doc.metadata.get("year", "UNKNOWN")
        page=doc.metadata.get("page", "UNKNOWN")

        formatted.append(f"[Source {i+1}: {ticker} {year} 10-K, Page {page}]\n"
                         f"{doc.page_content.strip()}")
        
        return "\n\n--\n\n".join(formatted)