"""Benchmark 3 retrieval strategies (Dense, BM25, Hybrid RRF) on Dev and Test splits."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, pack_context


def hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """Return True if chunk matches source, page, and contains gold evidence."""
    return any(
        chunk.get("source_id") == g.get("source_id")
        and chunk.get("page") == g.get("page")
        and g.get("evidence", "") in chunk.get("text", "")
        for g in gold
    )


def evaluate_retriever(
    retriever: Any,
    questions: list[dict[str, Any]],
    top_k: int = 3,
    budget: int = 800,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run retrieval and compute Hit@1, Hit@3, Context Hit, latency."""
    traces = []
    for q in questions:
        start_t = time.perf_counter()
        filters = q.get("filter")
        results = retriever.search(q["question"], filters=filters, top_k=top_k)
        context, ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        marked = [{**c, "hit": hit(c, q["gold"])} for c in results]
        hit1 = bool(marked and marked[0]["hit"])
        hit3 = any(c["hit"] for c in marked)
        context_hit = any(c["hit"] and c["id"] in ids for c in marked)

        traces.append({
            "query_id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "strategy": getattr(retriever, "strategy_name", "retrieval"),
            "top_k": top_k,
            "budget_chars": budget,
            "gold": q["gold"],
            "filter": filters,
            "results": marked,
            "context": context,
            "context_ids": ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "hit1": hit1,
            "hit3": hit3,
            "context_hit": context_hit,
        })

    n = len(traces)
    summary = {
        "strategy": getattr(retriever, "strategy_name", "retrieval"),
        "split": questions[0]["split"] if questions else "unknown",
        "n": n,
        "hit1": sum(t["hit1"] for t in traces),
        "hit1_rate": round(sum(t["hit1"] for t in traces) / n, 4) if n else 0.0,
        "hit3": sum(t["hit3"] for t in traces),
        "hit3_rate": round(sum(t["hit3"] for t in traces) / n, 4) if n else 0.0,
        "context_hit": sum(t["context_hit"] for t in traces),
        "context_hit_rate": round(sum(t["context_hit"] for t in traces) / n, 4) if n else 0.0,
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1) if n else 0.0,
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2) if n else 0.0,
        "service_cost_usd": 0.0,
    }
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval strategies.")
    parser.add_argument("--chunks", default="data/converted/chunks.json")
    parser.add_argument("--questions", default="data/converted/questions.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--output_dir", default="results/retrieval")
    parser.add_argument("--budget", type=int, default=800)
    args = parser.parse_args()

    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    all_questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = [q for q in all_questions if q.get("split") == args.split]

    print(f"=== Benchmarking Retrieval on Split: {args.split.upper()} (N={len(questions)}) ===")

    dense = DenseRetriever(chunks)
    dense.strategy_name = "dense_retrieval"  # type: ignore[attr-defined]

    bm25 = BM25Retriever(chunks)
    bm25.strategy_name = "bm25_lexical"  # type: ignore[attr-defined]

    hybrid = HybridRetriever(dense, bm25)
    hybrid.strategy_name = "hybrid_dense_bm25_rrf"  # type: ignore[attr-defined]

    retrievers = [dense, bm25, hybrid]
    results_summary = []

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for r in retrievers:
        summary, traces = evaluate_retriever(r, questions, top_k=3, budget=args.budget)
        results_summary.append(summary)

        # Save individual traces
        trace_file = out_path / f"traces_{summary['strategy']}_{args.split}.json"
        trace_file.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")

        print(
            f"[{summary['strategy']}] "
            f"Hit@1: {summary['hit1']}/{summary['n']} ({summary['hit1_rate']*100:.1f}%) | "
            f"Hit@3: {summary['hit3']}/{summary['n']} ({summary['hit3_rate']*100:.1f}%) | "
            f"Context: {summary['context_hit']}/{summary['n']} ({summary['context_hit_rate']*100:.1f}%) | "
            f"Latency: {summary['median_ms']} ms"
        )

    summary_file = out_path / f"summary_comparison_{args.split}.json"
    summary_file.write_text(json.dumps(results_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved all results and traces to {out_path}/")


if __name__ == "__main__":
    main()
