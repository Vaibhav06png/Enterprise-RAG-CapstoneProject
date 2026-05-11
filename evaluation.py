# =====================================================
# evaluation.py
# -----------------------------------------------------
# Compares two systems on a few sample customer queries:
#
#   (A) Baseline RAG         -> just retrieval + LLM
#   (B) Agentic RAG (CrewAI) -> 4-agent workflow
#
# Metrics:
#   * ROUGE-1, ROUGE-L (overlap with the reference answer)
#   * Relevance score (very simple: keyword overlap with retrieved chunks)
# =====================================================

import pandas as pd
from rouge_score import rouge_scorer

from app.rag import rag_pipeline
from app.agents.agents import run_agentic_workflow


# In a real project you'd sample ~50-100 rows from the CSV.
EVAL_SAMPLES = [
    {
        "query": "How do I cancel my order?",
        "reference": "You can cancel your order from the Orders page in your account, "
                     "as long as it has not shipped yet.",
    },
    {
        "query": "I want to get a refund for a damaged product.",
        "reference": "Please request a refund through the Returns section. "
                     "Damaged items are eligible for a full refund.",
    },
    {
        "query": "How can I update my shipping address?",
        "reference": "Go to your account settings and edit your shipping address before "
                     "the order is dispatched.",
    },
]


def baseline_rag(query: str):
    """Plain RAG: retrieval + grounded LLM answer. No agents."""
    return rag_pipeline.generate_answer(query)


def simple_relevance(answer: str, sources: list) -> float:
    """
    Very simple relevance metric: how many words in the answer
    also appear in the retrieved sources. 0.0 - 1.0.
    """
    if not answer or not sources:
        return 0.0
    ans_words = set(answer.lower().split())
    src_text = " ".join(s["content"].lower() for s in sources)
    src_words = set(src_text.split())
    overlap = ans_words & src_words
    return round(len(overlap) / max(len(ans_words), 1), 3)


def evaluate():
    # Build RAG once
    rag_pipeline.build_rag_pipeline()

    # ROUGE scorer (measures overlap of n-grams with the reference)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rows = []
    for sample in EVAL_SAMPLES:
        q, ref = sample["query"], sample["reference"]
        print(f"\n=== Query: {q} ===")

        # --- Baseline RAG ---
        base = baseline_rag(q)
        base_scores = scorer.score(ref, base["answer"])
        base_rel = simple_relevance(base["answer"], base["sources"])

        # --- Agentic ---
        agent = run_agentic_workflow(q)
        agent_scores = scorer.score(ref, agent["final_response"])
        agent_rel = simple_relevance(agent["final_response"], agent["retrieved_sources"])

        rows.append({
            "query": q,
            "baseline_rouge1": round(base_scores["rouge1"].fmeasure, 3),
            "baseline_rougeL": round(base_scores["rougeL"].fmeasure, 3),
            "baseline_relevance": base_rel,
            "agentic_rouge1": round(agent_scores["rouge1"].fmeasure, 3),
            "agentic_rougeL": round(agent_scores["rougeL"].fmeasure, 3),
            "agentic_relevance": agent_rel,
            "escalated": agent["escalate_flag"],
        })

    df = pd.DataFrame(rows)
    print("\n=== EVALUATION RESULTS ===")
    print(df.to_string(index=False))

    print("\n=== AVERAGES ===")
    print(df.drop(columns=["query", "escalated"]).mean(numeric_only=True).round(3))

    df.to_csv("evaluation_results.csv", index=False)
    print("\nSaved -> evaluation_results.csv")


if __name__ == "__main__":
    evaluate()
