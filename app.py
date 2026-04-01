import streamlit as st
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve()))

# ── Page config (must be first Streamlit call) ────────────────────
st.set_page_config(
    page_title="FinSight RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────────────
from src.ingestion.loader import load_all_pdfs
from src.ingestion.chunker import chunk_documents
from src.retrieval.vectorstore import build_vectorstore, load_vectorstore
from src.retrieval.retriever import build_hybrid_retriever
from src.retrieval.reranker import build_reranker
from src.generation.chain import build_rag_chain_with_sources
from config.settings import DATA_RAW_DIR, FAISS_DIR, CHUNK_STRATEGY


# ── Pipeline loader (cached so it only runs once) ─────────────────
# @st.cache_resource tells Streamlit: run this function once,
# then reuse the result for every user interaction.
# Without this, the entire pipeline would reload on every button click.

@st.cache_resource(show_spinner="Loading pipeline... (first run takes ~60s)")
def load_pipeline():
    """Load the full RAG pipeline once and cache it."""
    pages  = load_all_pdfs(DATA_RAW_DIR)
    chunks = chunk_documents(pages, strategy=CHUNK_STRATEGY)

    if Path(FAISS_DIR).exists():
        vectorstore = load_vectorstore()
    else:
        vectorstore = build_vectorstore(chunks)

    hybrid_retriever    = build_hybrid_retriever(vectorstore, chunks)
    reranking_retriever = build_reranker(hybrid_retriever)
    rag_chain           = build_rag_chain_with_sources(reranking_retriever)

    return rag_chain, chunks


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 FinSight RAG")
    st.caption("SEC 10-K Financial Filing Q&A")
    st.divider()

    st.subheader("Configuration")
    selected_strategy = st.selectbox(
        "Chunking strategy",
        ["recursive", "fixed", "semantic"],
        index=0,
        help="Strategy used to split documents into chunks"
    )

    top_k = st.slider(
        "Top-K chunks",
        min_value=1, max_value=10, value=5,
        help="Number of chunks retrieved per query"
    )

    st.divider()

    # Show available filings
    st.subheader("Loaded filings")
    data_dir = Path(DATA_RAW_DIR)
    files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.htm"))
    if files:
        for f in sorted(files):
            st.markdown(f"📄 `{f.name}`")
    else:
        st.warning("No filings found in data/raw/")

    st.divider()
    st.caption("Built with LangChain · FAISS · Mistral · RAGAS")


# ── Main tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Q&A", "🔍 Retrieved Chunks", "📈 Evaluation"])


# TAB 1 — Q&A Interface
with tab1:
    st.header("Ask questions about SEC filings")

    # Example questions to help users get started
    st.markdown("**Try asking:**")
    example_cols = st.columns(3)
    examples = [
        "What was Apple's total revenue in 2025?",
        "What are Amazon's main business segments?",
        "What risk factors does JPMorgan mention?",
    ]
    for col, example in zip(example_cols, examples):
        if col.button(example, use_container_width=True):
            st.session_state["query"] = example

    st.divider()

    # Query input
    query = st.text_input(
        "Your question",
        value=st.session_state.get("query", ""),
        placeholder="e.g. What was Apple's net income in 2025?",
        key="query_input",
    )

    ask_btn = st.button("Ask", type="primary", use_container_width=False)

    if ask_btn and query:
        # Load pipeline
        with st.spinner("Loading pipeline..."):
            rag_chain, chunks = load_pipeline()

        # Run query
        with st.spinner("Searching filings and generating answer..."):
            try:
                result = rag_chain.invoke({"question": query})
                answer      = result["answer"]
                source_docs = result["source_docs"]

                # Save to session state so Tab 2 can display chunks
                st.session_state["last_answer"]      = answer
                st.session_state["last_source_docs"] = source_docs
                st.session_state["last_query"]       = query

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure Ollama is running: `ollama serve`")
                st.stop()

        # Display answer
        st.subheader("Answer")
        st.markdown(answer)

        # Source summary below answer
        st.divider()
        st.caption(f"Answer generated from {len(source_docs)} retrieved chunks")
        source_summary = set()
        for doc in source_docs:
            ticker = doc.metadata.get("ticker", "?")
            year   = doc.metadata.get("year", "?")
            source_summary.add(f"{ticker} {year} 10-K")

        st.markdown("**Sources used:** " + " · ".join(sorted(source_summary)))

    elif not query and ask_btn:
        st.warning("Please enter a question.")


# TAB 2 — Retrieved Chunks Inspector
with tab2:
    st.header("Retrieved chunks inspector")
    st.caption(
        "See exactly which document chunks were used to generate the last answer. "
        "This is what makes RAG answers auditable."
    )

    if "last_source_docs" not in st.session_state:
        st.info("Ask a question in the Q&A tab first.")
    else:
        source_docs = st.session_state["last_source_docs"]
        query       = st.session_state.get("last_query", "")

        st.markdown(f"**Query:** {query}")
        st.markdown(f"**Chunks retrieved:** {len(source_docs)}")
        st.divider()

        for i, doc in enumerate(source_docs):
            ticker   = doc.metadata.get("ticker", "UNKNOWN")
            year     = doc.metadata.get("year", "UNKNOWN")
            page     = doc.metadata.get("page", "?")
            strategy = doc.metadata.get("chunk_strategy", "?")
            chunk_id = doc.metadata.get("chunk_id", "?")

            with st.expander(
                f"Chunk {i+1} — {ticker} {year} 10-K · Page {page}",
                expanded=(i == 0),  
            ):
                # Metadata pills
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Company", ticker)
                col2.metric("Year", year)
                col3.metric("Page", page)
                col4.metric("Strategy", strategy)

                st.divider()

                # Chunk text
                st.markdown("**Chunk content:**")
                st.text_area(
                    label="",
                    value=doc.page_content,
                    height=200,
                    key=f"chunk_{i}",
                    disabled=True,
                )


# TAB 3 — Evaluation Dashboard
with tab3:
    st.header("RAGAS Evaluation Dashboard")
    st.caption(
        "Compare chunking strategies using RAGAS metrics. "
        "Run the evaluation once from the terminal, then view results here."
    )

    # Look for saved evaluation results
    processed_dir = Path("data/processed")
    result_files  = sorted(processed_dir.glob("ragas_results_*.csv")) \
                    if processed_dir.exists() else []

    if not result_files:
        st.info(
            "No evaluation results found yet. Run the evaluation first:\n\n"
            "```bash\npython main.py --mode evaluate\n```"
        )
    else:
        # Load the most recent results file
        latest = result_files[-1]
        df = pd.read_csv(latest)

        st.success(f"Loaded results from `{latest.name}`")
        st.divider()

        # ── Summary table ──────────────────────────────────────────
        metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

        # Only keep metrics that exist in the dataframe
        available = [m for m in metrics if m in df.columns]

        summary = df.groupby("strategy")[available].mean().round(3)
        summary["avg_score"] = summary[available].mean(axis=1).round(3)
        summary = summary.sort_values("avg_score", ascending=False)

        st.subheader("Strategy comparison")

        # Highlight the best strategy
        best = summary.index[0]
        st.success(f"Best performing strategy: **{best}** chunking "
                   f"(avg score: {summary.loc[best, 'avg_score']:.3f})")

        st.dataframe(
            summary.style.highlight_max(axis=0, color="#d4edda"),
            use_container_width=True,
        )

        st.divider()

        # ── Per-metric bar charts ──────────────────────────────────
        st.subheader("Metric breakdown")

        chart_cols = st.columns(2)
        for i, metric in enumerate(available):
            with chart_cols[i % 2]:
                chart_data = summary[[metric]].reset_index()
                st.markdown(f"**{metric.replace('_', ' ').title()}**")
                st.bar_chart(
                    chart_data.set_index("strategy")[metric],
                    use_container_width=True,
                    height=200,
                )

        st.divider()

        # ── Per-question breakdown ─────────────────────────────────
        st.subheader("Per-question breakdown")

        strategy_filter = st.selectbox(
            "Filter by strategy",
            options=["all"] + list(df["strategy"].unique()),
        )

        filtered_df = df if strategy_filter == "all" \
                      else df[df["strategy"] == strategy_filter]

        display_cols = ["question", "strategy"] + available
        display_cols = [c for c in display_cols if c in filtered_df.columns]

        st.dataframe(
            filtered_df[display_cols].round(3),
            use_container_width=True,
            height=400,
        )