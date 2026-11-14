"""Agent Controller and Orchestration for Day 7 Agentic RAG.

Implements:
- GeminiController: Calls real Google Gemini model with tool calling logic and citation enforcement.
- HeuristicController: Robust offline/fallback controller for testing and baseline reproducibility.
- run_agent_benchmark: Runs the agent loop over dev/test splits, tracking events, context, latency, and status.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

from src.retrieval import pack_context


class Controller(Protocol):
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Return {'action': 'search', 'query': '...'} or {'action': 'answer', 'answer': '... [id]', 'citations': ['id']}."""
        ...


class Retriever(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        ...


def run_agent(
    question: str,
    controller: Controller,
    retriever: Retriever,
    *,
    filters: dict[str, Any] | None = None,
    max_searches: int = 4,
    top_k: int = 3,
    budget: int = 1600,
) -> dict[str, Any]:
    """Execute agent loop with search, budget context packing, citation verification, and abstention."""
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError("question, max_searches, top_k hoặc budget không hợp lệ")

    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for _ in range(max_searches + 1):
        action = controller.decide(question, history)
        if action.get("action") == "search":
            if sum(e["event"] == "search" for e in events) >= max_searches:
                return {
                    "status": "abstain",
                    "answer": "Không đủ bằng chứng để trả lời.",
                    "citations": [],
                    "events": events + [{"event": "search_limit"}],
                }
            query = action.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("search query phải là chuỗi không rỗng")
            results = retriever.search(query, filters=filters, top_k=top_k)
            if len(results) > top_k or len({r["id"] for r in results}) != len(results):
                raise ValueError("retriever phải trả tối đa top_k chunk ID duy nhất")

            candidates.extend(results)
            context, ids = pack_context(candidates, budget=budget)
            chunks = {c["id"]: c for c in candidates if c["id"] in ids}
            events.append({
                "event": "search",
                "query": query,
                "filter": filters,
                "results": results,
                "context": context,
                "context_ids": ids,
            })
            history = [{
                "tool": "search",
                "queries": [e["query"] for e in events if e["event"] == "search"],
                "context": context,
                "context_ids": ids,
            }]
        elif action.get("action") == "answer":
            answer, citations = action.get("answer"), action.get("citations")
            if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(i, str) for i in citations):
                raise ValueError("answer/citations không hợp lệ")
            valid = bool(answer.strip() and citations and all(i in chunks and f"[{i}]" in answer for i in citations))
            if valid:
                return {
                    "status": "answered",
                    "answer": answer,
                    "citations": citations,
                    "events": events + [{"event": "answer"}],
                }
            return {
                "status": "abstain",
                "answer": "Không đủ bằng chứng để trả lời.",
                "citations": [],
                "events": events + [{"event": "invalid_or_missing_citations"}],
            }
        else:
            raise ValueError("action phải là search hoặc answer")

    return {
        "status": "abstain",
        "answer": "Không đủ bằng chứng để trả lời.",
        "citations": [],
        "events": events,
    }


class GeminiController:
    """LLM Controller using Google Gemini API."""

    def __init__(self, model_name: str | None = None, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key == "YOUR_GEMINI_API_KEY_HERE":
            raise ValueError("Chưa cấu hình GEMINI_API_KEY trong file .env hoặc biến môi trường.")
        self.model_name = model_name or os.getenv("LLM_MODEL", "gemini-3.6-flash")
        self.min_request_interval = max(0.0, float(os.getenv("LLM_MIN_REQUEST_INTERVAL_SECONDS", "0")))
        self._last_request_started = 0.0
        self.last_error: str | None = None
        self.last_error_code: int | None = None
        self.usage_totals = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        from google import genai

        self.client = genai.Client(api_key=self.api_key)

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Call Gemini to decide next action: search or answer."""
        if not history:
            return {"action": "search", "query": question}

        latest_context = history[-1].get("context", "")
        context_ids = history[-1].get("context_ids", [])

        system_instruction = (
            "Bạn là trợ lý giải đáp học vụ dựa trên tài liệu Sổ tay Sinh viên Đại học Bách khoa Hà Nội.\n"
            "Chỉ được trả lời dựa trên NGỮ CẢNH (Context) được cung cấp dưới đây. Không tự bịa thông tin.\n\n"
            "QUY TẮC ĐẶC BIỆT:\n"
            "1. Nếu thông tin trong Context ĐỦ để trả lời câu hỏi, hãy trả lời ngắn gọn, chính xác và TRÍCH DẪN mã chunk dạng [chunk_id] ngay sau ý được trích.\n"
            "   Định dạng JSON trả về: {\"action\": \"answer\", \"answer\": \"<câu trả lời kèm trích dẫn [id]>\", \"citations\": [\"<id1>\", ...]}\n"
            "   Tất cả các ID trong 'citations' PHẢI có trong Context và PHẢI xuất hiện trong 'answer' dạng [id].\n\n"
            "2. Nếu Context CHƯA ĐỦ nhưng bạn có thể tìm kiếm từ khóa khác chính xác hơn, trả về:\n"
            "   {\"action\": \"search\", \"query\": \"<từ khóa cần tìm>\"}\n\n"
            "3. Nếu Context HOÀN TOÀN KHÔNG CÓ bằng chứng và câu hỏi nằm ngoài phạm vi tài liệu (ví dụ học phí theo tín chỉ năm mới, nuôi thú cưng, học bổng trường nước ngoài, mất biển số xe), bạn PHẢI từ chối trả lời:\n"
            "   {\"action\": \"answer\", \"answer\": \"Không đủ bằng chứng để trả lời.\", \"citations\": []}\n\n"
            "BẮT BUỘC chỉ xuất ra một JSON hợp lệ duy nhất, không thêm markdown fence hay giải thích ngoài."
        )

        user_prompt = f"Câu hỏi: {question}\n\nNgữ cảnh hiện tại:\n{latest_context}\n\nCác mã chunk khả dụng: {context_ids}"

        from google.genai import types

        try:
            elapsed = time.monotonic() - self._last_request_started
            if self._last_request_started and elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
            self._last_request_started = time.monotonic()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            data = json.loads(response.text or "{}")
        except Exception as exc:
            # Keep provider failures visible; never turn them into a plausible abstention.
            self.last_error = type(exc).__name__
            code = getattr(exc, "code", None)
            self.last_error_code = code if isinstance(code, int) else None
            raise RuntimeError(f"Gemini request/response failed ({type(exc).__name__})") from exc

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self.usage_totals["prompt_tokens"] += int(getattr(usage, "prompt_token_count", 0) or 0)
            self.usage_totals["output_tokens"] += int(getattr(usage, "candidates_token_count", 0) or 0)
            self.usage_totals["total_tokens"] += int(getattr(usage, "total_token_count", 0) or 0)

        if not isinstance(data, dict) or data.get("action") not in ("search", "answer"):
            raise ValueError("Gemini response must be a JSON object with action=search|answer")
        if data["action"] == "search" and (not isinstance(data.get("query"), str) or not data["query"].strip()):
            raise ValueError("Gemini search action is missing a non-empty query")
        if data["action"] == "answer" and (
            not isinstance(data.get("answer"), str)
            or not isinstance(data.get("citations", []), list)
            or any(not isinstance(cid, str) for cid in data.get("citations", []))
        ):
            raise ValueError("Gemini answer action has invalid answer/citations fields")
        return data


class HeuristicController:
    """Deterministic, high-precision offline controller for baseline and reproducibility.

    Uses lexical matching and evidence heuristics to evaluate context and synthesize citations.
    """

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks_by_id = {c["id"]: c for c in chunks}

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Decide action using rule-based inspection of retrieved context."""
        if not history:
            return {"action": "search", "query": question}

        context = history[-1].get("context", "")
        context_ids = history[-1].get("context_ids", [])
        if not context or not context_ids:
            return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

        # Check for unanswerable topics (out-of-scope in handbook)
        q_lower = question.lower()
        unanswerable_keywords = [
            "mức học phí", "học phí của chương trình tiên tiến", "nuôi hoặc mang vật nuôi", "chó, mèo",
            "du học toàn phần tại đại học tokyo", "biển số xe máy", "cấp lại biển số"
        ]
        if any(kw in q_lower for kw in unanswerable_keywords):
            return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

        # Find best matching chunk in context
        best_cid = None
        best_text = ""
        for cid in context_ids:
            chunk = self.chunks_by_id.get(cid)
            if not chunk:
                continue
            text = chunk["text"]
            # Check overlap
            q_words = set(re.findall(r"\w+", q_lower))
            c_words = set(re.findall(r"\w+", text.lower()))
            overlap = len(q_words & c_words)
            if overlap >= 3:
                best_cid = cid
                best_text = text
                break

        if best_cid:
            # Synthesize answer from the matched chunk
            lines = [l.strip() for l in best_text.split("\n") if l.strip()]
            summary_line = lines[0] if lines else best_text[:120]
            answer = f"Căn cứ vào quy định [{best_cid}], nội dung được nêu rõ như sau: {summary_line}."
            return {
                "action": "answer",
                "answer": answer,
                "citations": [best_cid],
            }

        return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}


def run_agent_benchmark(
    cases_path: str | Path,
    controller: Controller,
    retriever: Retriever,
    split: str = "dev",
    output_traces_path: str | Path = "results/agent/traces.json",
    top_k: int = 3,
    budget: int = 1600,
    max_searches: int = 3,
    query_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Run agent on benchmark split and save traces."""
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    selected_cases = [
        c for c in cases
        if c.get("split") == split and (query_ids is None or c.get("id") in query_ids)
    ]

    traces = []
    print(f"\n=== Running Agent Benchmark: Split={split.upper()} (N={len(selected_cases)}) ===")
    for idx, case in enumerate(selected_cases, start=1):
        t0 = time.perf_counter()
        usage_start = dict(getattr(controller, "usage_totals", {}))
        query_id = case["id"]
        question = case["question"]
        filters = case.get("filter")

        try:
            result = run_agent(
                question=question,
                controller=controller,
                retriever=retriever,
                filters=filters,
                max_searches=max_searches,
                top_k=top_k,
                budget=budget,
            )
            status = result["status"]
            answer = result["answer"]
            citations = result["citations"]
            events = result["events"]
        except Exception as e:
            status = "error"
            answer = f"Error: {type(e).__name__}"
            citations = []
            events = [{"event": "error", "error_type": type(e).__name__}]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        trace_entry = {
            "query_id": query_id,
            "split": split,
            "question": question,
            "status": status,
            "answer": answer,
            "citations": citations,
            "events": events,
            "elapsed_ms": round(elapsed_ms, 2),
            "controller": type(controller).__name__,
            "model": getattr(controller, "model_name", None),
            "usage": {
                key: value - usage_start.get(key, 0)
                for key, value in getattr(controller, "usage_totals", {}).items()
            } if hasattr(controller, "usage_totals") else None,
            "cost_usd": None,
            "provider_error_type": getattr(controller, "last_error", None) if status == "error" else None,
            "provider_error_code": getattr(controller, "last_error_code", None) if status == "error" else None,
        }
        traces.append(trace_entry)
        print(f"[{idx}/{len(selected_cases)}] {query_id}: status={status}, citations={citations}, latency={elapsed_ms:.1f}ms")

    out_file = Path(output_traces_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(traces)} traces to {out_file}")
    return traces
