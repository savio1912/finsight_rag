# src/evaluation/ragas_eval.py

from pathlib import Path
from typing import List, Dict
import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

import sys
from config.settings import DATA_RAW_DIR, FAISS_DIR, CHUNK_STRATEGY
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.settings import LLM_MODEL, EMBEDDING_MODEL


def build_ragas_judges():
    """
    RAGAS needs an LLM and embedding model to judge answers.
    We use the same local Mistral + HuggingFace setup — no OpenAI needed.

    Why do we need judges?
    - Faithfulness: RAGAS asks the judge LLM to verify each claim
      in the answer against the retrieved context
    - Answer relevancy: RAGAS generates reverse questions from the
      answer and checks if they match the original question
    - These require an LLM to "understand" text, not just count words
    """
    judge_llm = LangchainLLMWrapper(
        ChatOllama(model=LLM_MODEL, temperature=0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )
    return judge_llm, judge_embeddings


def run_pipeline_on_testset(
    rag_chain,
    test_set: List[Dict],
) -> List[Dict]:
    """
    Run each question through the full RAG pipeline and collect:
    - question
    - generated answer
    - retrieved contexts (as list of strings)
    - ground truth answer

    This is the data RAGAS needs to compute all four metrics.
    """
    results = []

    print(f"\nRunning pipeline on {len(test_set)} test questions...")

    for i, item in enumerate(test_set):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i+1}/{len(test_set)}] {question[:60]}...")

        try:
            # Run the full RAG chain
            output = rag_chain.invoke({"question": question})

            answer = output["answer"]
            source_docs = output["source_docs"]

            # RAGAS expects contexts as a list of strings
            contexts = [doc.page_content for doc in source_docs]

            results.append({
                "question":     question,
                "answer":       answer,
                "contexts":     contexts,
                "ground_truth": ground_truth,
            })

        except Exception as e:
            print(f"    Error on question {i+1}: {e}")
            # Add a failed entry so we don't lose track
            results.append({
                "question":     question,
                "answer":       "ERROR",
                "contexts":     [],
                "ground_truth": ground_truth,
            })

    print(f"Pipeline run complete. {len(results)} results collected.")
    return results


def evaluate_with_ragas(results: List[Dict], strategy_name: str) -> pd.DataFrame:
    """
    Feed results into RAGAS and get scores for all four metrics.

    Returns a DataFrame with one row per question and columns
    for each metric — easy to compare across chunking strategies.
    """
    print(f"\nRunning RAGAS evaluation for strategy: '{strategy_name}'...")

    # RAGAS works with HuggingFace Dataset format
    dataset = Dataset.from_list(results)

    judge_llm, judge_embeddings = build_ragas_judges()

    # Run evaluation — this takes a few minutes
    # RAGAS calls the judge LLM once per metric per question
    scores = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    # Convert to DataFrame
    df = scores.to_pandas()
    df["strategy"] = strategy_name

    # Print summary
    print(f"\n{'='*50}")
    print(f"RAGAS Results — Strategy: {strategy_name}")
    print(f"{'='*50}")
    print(f"  Faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"  Answer Relevancy:  {df['answer_relevancy'].mean():.3f}")
    print(f"  Context Precision: {df['context_precision'].mean():.3f}")
    print(f"  Context Recall:    {df['context_recall'].mean():.3f}")
    print(f"{'='*50}")

    return df


def compare_chunking_strategies(
    pipeline_builder_fn,
    test_set: List[Dict],
) -> pd.DataFrame:
    """
    Run the full evaluation across all three chunking strategies
    and return a combined DataFrame for comparison.

    This is the key function that produces the comparison table
    shown in the README and the Streamlit evaluation tab.

    pipeline_builder_fn: a function that takes a strategy name
    and returns a ready-to-use rag_chain
    """
    all_results = []

    for strategy in ["fixed", "recursive", "semantic"]:
        print(f"\n{'='*50}")
        print(f"Evaluating chunking strategy: {strategy}")
        print(f"{'='*50}")

        # Build a fresh pipeline with this chunking strategy
        rag_chain = pipeline_builder_fn()

        # Run all test questions
        results = run_pipeline_on_testset(rag_chain, test_set)

        # Score with RAGAS
        df = evaluate_with_ragas(results, strategy_name=strategy)
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    return combined