from pathlib import Path
import argparse
from src.generation import chain
from src.generation.chain import build_rag_chain_with_sources
from src.ingestion.loader import load_all_pdfs
from src.ingestion.chunker import chunk_documents
from src.retrieval.vectorstore import build_vectorstore, load_vectorstore
from src.retrieval.retriever import build_hybrid_retriever, retrieve
from src.retrieval.reranker import build_reranker
from src.evaluation.testset import TEST_SET
from src.evaluation.ragas_eval import compare_chunking_strategies
from src.evaluation.metrics import save_results, print_comparison_table
from config.settings import DATA_RAW_DIR, FAISS_DIR, CHUNK_STRATEGY

FORCE_REBUILD = False

def build_pipeline():
    # ── Step 1: Check if vector store already exists ──
    if not FORCE_REBUILD and Path(FAISS_DIR).exists():
        print("Loading existing vector store...")
        vectorstore = load_vectorstore()
        pages=load_all_pdfs(DATA_RAW_DIR)
        chunks =chunk_documents(pages, strategy=CHUNK_STRATEGY)

    else:
        #step 2: Ingest
        print('Building vector store from scratch...')
        pages=load_all_pdfs(DATA_RAW_DIR)
        chunks =chunk_documents(pages, strategy=CHUNK_STRATEGY)

        #step 3: Build vector store
        vectorstore = build_vectorstore(chunks)

    #step 4: Build retriever
    hybrid_retriever = build_hybrid_retriever(vectorstore, chunks)
    reranking_retriever = build_reranker(hybrid_retriever)

    rag_chain=build_rag_chain_with_sources(reranking_retriever)
    return rag_chain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["query", "evaluate"],
        default="query",
        help="query: ask questions | evaluate: run RAGAS benchmark"
    )
    args = parser.parse_args()
    # rag_chain = build_pipeline()
    if args.mode == "query":
        print('Query mode')
        # Normal Q&A mode
        chain = build_pipeline()

        test_queries = [
            "What was Apple's total revenue in 2023?",
            "What are the main risk factors mentioned?",
            "How did operating income change year over year?",
        ]

        for query in test_queries:
            print(f"\nQuery: {query}")
            result = chain.invoke({"question": query})
            print(f"\nQ: {query}")
            print(f"A: {result['answer']}")
            print(f"Sources used: {len(result['source_docs'])} chunks")
            print("-" * 60)
    elif args.mode == "evaluate":
        print('Evaluate mode')
        # RAGAS evaluation mode — compares all 3 chunking strategies
        print("Starting RAGAS evaluation across chunking strategies...")
        print("This will take 15-30 minutes depending on your hardware.\n")

        combined_df = compare_chunking_strategies(
            pipeline_builder_fn=build_pipeline,
            test_set=TEST_SET,
        )

        save_results(combined_df)
        print_comparison_table(combined_df)
