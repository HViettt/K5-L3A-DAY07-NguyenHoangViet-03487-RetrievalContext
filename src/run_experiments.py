"""Main experiment orchestration script for Day 7 Lab.

Runs:
1. Retrieval benchmarks on Dev & Test across all configurations:
   - Baseline: BM25 (budget=800, top_k=3)
   - Phương pháp 2: Dense Retrieval (budget=800, top_k=3)
   - Phương pháp 3: Hybrid RRF (budget=800, top_k=3)
   - Cải thiện 1: Dense Retrieval với budget mở rộng (budget=1600, top_k=3)
   - Cải thiện 2: Hybrid RRF với budget mở rộng (budget=1800, top_k=3)
   - Chốt cấu hình tối ưu: Hybrid RRF (budget=1800)
   - Đánh giá Cấu hình đã chọn vs Baseline trên Test split.

2. Agent benchmarks on Dev & Test:
   - Executes agent loop over 24 cases.
   - Saves complete traces.
   - Performs automated and audited review validation.
   - Computes summary metrics with starter/agent_metrics.py.
"""
from __future__ import annotations

import json
import os
import argparse
import subprocess
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import GeminiController, HeuristicController, run_agent_benchmark
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_retrieval_suite():
    print("=" * 60)
    print("PHẦN 1: ĐO LƯỜNG VÀ SO SÁNH CÁC PHƯƠNG PHÁP RETRIEVAL")
    print("=" * 60)

    chunks = json.loads(Path("data/converted/chunks.json").read_text(encoding="utf-8"))
    questions = json.loads(Path("data/converted/questions.json").read_text(encoding="utf-8"))

    dense = DenseRetriever(chunks)
    dense.strategy_name = "dense_retrieval"

    bm25 = BM25Retriever(chunks)
    bm25.strategy_name = "bm25_lexical"

    hybrid = HybridRetriever(dense, bm25)
    hybrid.strategy_name = "hybrid_rrf"

    from src.benchmark_retrieval import evaluate_retriever

    configs = [
        {"name": "Baseline (BM25)", "trace": "traces_bm25_dev.json", "retriever": bm25, "budget": 800, "split": "dev"},
        {"name": "Phương pháp 2 (Dense)", "trace": "traces_dense_dev.json", "retriever": dense, "budget": 800, "split": "dev"},
        {"name": "Phương pháp 3 (Hybrid RRF)", "trace": "traces_hybrid_dev.json", "retriever": hybrid, "budget": 800, "split": "dev"},
        {"name": "Cải thiện 1 (Dense + Budget 1600)", "trace": "traces_dense_budget1600_dev.json", "retriever": dense, "budget": 1600, "split": "dev"},
        {"name": "Cải thiện 2 (Hybrid RRF + Budget 1800)", "trace": "traces_hybrid_budget1800_dev.json", "retriever": hybrid, "budget": 1800, "split": "dev"},
        {"name": "Cấu hình đã chọn (Hybrid RRF + Budget 1800)", "trace": "traces_selected_hybrid_test.json", "retriever": hybrid, "budget": 1800, "split": "test"},
        {"name": "Baseline trên Test (BM25 + Budget 800)", "trace": "traces_baseline_bm25_test.json", "retriever": bm25, "budget": 800, "split": "test"},
    ]

    retrieval_records = []
    out_dir = Path("results/retrieval")
    out_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        split_qs = [q for q in questions if q.get("split") == cfg["split"]]
        summary, traces = evaluate_retriever(cfg["retriever"], split_qs, top_k=3, budget=cfg["budget"])
        (out_dir / cfg["trace"]).write_text(
            json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        record = {
            "config_name": cfg["name"],
            "strategy": summary["strategy"],
            "split": cfg["split"],
            "budget": cfg["budget"],
            "n": summary["n"],
            "hit1": summary["hit1"],
            "hit1_rate": summary["hit1_rate"],
            "hit3": summary["hit3"],
            "hit3_rate": summary["hit3_rate"],
            "context_hit": summary["context_hit"],
            "context_hit_rate": summary["context_hit_rate"],
            "median_ms": summary["median_ms"],
            "service_cost_usd": 0.0,
        }
        retrieval_records.append(record)
        print(
            f"[{cfg['name']} - {cfg['split'].upper()}] "
            f"Hit@1: {summary['hit1']}/{summary['n']} ({summary['hit1_rate']*100:.1f}%) | "
            f"Hit@3: {summary['hit3']}/{summary['n']} ({summary['hit3_rate']*100:.1f}%) | "
            f"Context: {summary['context_hit']}/{summary['n']} ({summary['context_hit_rate']*100:.1f}%) | "
            f"Latency: {summary['median_ms']} ms"
        )

    (out_dir / "full_retrieval_comparison.json").write_text(
        json.dumps(retrieval_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nĐã lưu kết quả retrieval vào {out_dir / 'full_retrieval_comparison.json'}")
    return retrieval_records


def build_review_template(cases_path: str, traces_path: str, output_reviews_path: str):
    """Create pending labels while retaining completed human reviews."""
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    traces = json.loads(Path(traces_path).read_text(encoding="utf-8"))
    out_file = Path(output_reviews_path)
    prior = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else []
    prior_by_id = {r["query_id"]: r for r in prior}
    trace_map = {t["query_id"]: t for t in traces}
    reviews = []

    for case in cases:
        cid = case["id"]
        if cid not in trace_map:
            continue
        previous = prior_by_id.get(cid)
        if previous and previous.get("review_status") in {
            "human_reviewed_against_gold_and_cited_chunk",
            "assistant_reviewed_against_gold_and_cited_chunk",
        }:
            reviews.append(previous)
            continue
        t = trace_map[cid]
        citations = t.get("citations", [])
        review = {
            "query_id": cid,
            "citation_support": [None for _ in citations],
            "review_status": "pending_manual_review",
            "review_note": "Compare answer and every citation with the original evidence before assigning labels.",
        }
        if case["answerable"]:
            review["answer_grade"] = None
        else:
            review["abstention_correct"] = None
        reviews.append(review)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã tạo file audit review: {out_file} (N={len(reviews)})")
    return reviews


def run_agent_suite(offline: bool = False, retry_errors: bool = False):
    print("\n" + "=" * 60)
    print("PHẦN 2: THỰC THI VÀ ĐÁNH GIÁ AGENT (DEV & TEST)")
    print("=" * 60)

    chunks = json.loads(Path("data/converted/chunks.json").read_text(encoding="utf-8"))
    dense = DenseRetriever(chunks)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(dense, bm25)

    # Explicitly select live or offline mode. A failed live run must never be
    # silently replaced with heuristic answers and reported as a model result.
    if offline:
        controller = HeuristicController(chunks)
        output_dir = Path("results/agent_offline")
        print("Offline mode: HeuristicController (not valid as the required real-model evaluation).")
    else:
        controller = GeminiController()
        output_dir = Path("results/agent")
        print(f"Live model: {controller.model_name}")

    def run_config(split: str, retriever, trace_name: str, review_name: str, summary_name: str):
        trace_path = output_dir / trace_name
        review_path = output_dir / review_name
        summary_path = output_dir / summary_name
        if retry_errors and trace_path.exists():
            old = json.loads(trace_path.read_text(encoding="utf-8"))
            failed_ids = {t["query_id"] for t in old if t.get("status") == "error"}
            if not failed_ids:
                print(f"{trace_name}: no failed cases to retry")
                return old
            retry_path = output_dir / f".{trace_path.stem}_retry.json"
            retry_rows = run_agent_benchmark(
                cases_path="data/converted/cases.json",
                controller=controller,
                retriever=retriever,
                split=split,
                output_traces_path=retry_path,
                top_k=3,
                budget=1800,
                max_searches=3,
                query_ids=failed_ids,
            )
            retry_map = {t["query_id"]: t for t in retry_rows}
            merged = [retry_map.get(t["query_id"], t) for t in old]
            trace_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            retry_path.unlink(missing_ok=True)
            print(f"{trace_name}: retried {len(failed_ids)} failed case(s); retained completed traces")
        else:
            merged = run_agent_benchmark(
                cases_path="data/converted/cases.json",
                controller=controller,
                retriever=retriever,
                split=split,
                output_traces_path=trace_path,
                top_k=3,
                budget=1800,
                max_searches=3,
            )
        summary_path.unlink(missing_ok=True)
        build_review_template(
            cases_path="data/converted/cases.json",
            traces_path=str(trace_path),
            output_reviews_path=str(review_path),
        )
        return merged

    # 1. Run Dev
    dev_traces = run_config("dev", hybrid, "traces_dev.json", "reviews_dev.json", "summary_dev.json")
    if dev_traces and all(t.get("status") == "error" for t in dev_traces):
        raise RuntimeError("All dev requests failed; stopping before test. Check Gemini key, API access, and model availability.")

    # 2. Compare selected agent and BM25 baseline on the same test split.
    run_config("test", hybrid, "traces_test.json", "reviews_test.json", "summary_test.json")
    run_config("test", bm25, "traces_test_baseline.json", "reviews_test_baseline.json", "summary_test_baseline.json")

    print("Review templates saved. Read all answers/citations against source evidence, fill labels, then run starter/agent_metrics.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Day 7 retrieval and agent experiments")
    parser.add_argument("--retrieval-only", action="store_true", help="Run retrieval experiments without the agent")
    parser.add_argument("--offline-agent", action="store_true", help="Run labeled heuristic traces for debugging only")
    parser.add_argument("--retry-errors", action="store_true", help="Retry only saved provider-error cases and preserve completed traces/reviews")
    args = parser.parse_args()
    if args.retry_errors and args.offline_agent:
        parser.error("--retry-errors cannot be combined with --offline-agent")
    run_retrieval_suite()
    if not args.retrieval_only:
        try:
            run_agent_suite(offline=args.offline_agent, retry_errors=args.retry_errors)
        except (ValueError, RuntimeError) as exc:
            print(f"Agent run stopped: {exc}")
            raise SystemExit(2) from None
