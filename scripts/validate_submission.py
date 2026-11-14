"""Check the basic Day 7 submission data contract without calling external APIs."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    chunks = read_json("data/converted/chunks.json")
    questions = read_json("data/converted/questions.json")
    cases = read_json("data/converted/cases.json")
    cache = read_json("data/cache_embeddings.json")

    chunk_ids = [chunk["id"] for chunk in chunks]
    assert chunks and len(chunk_ids) == len(set(chunk_ids)), "chunk list is empty or IDs are duplicated"
    assert all(chunk.get(k) is not None for chunk in chunks for k in ("text", "source_id", "source", "page"))

    split_counts = Counter((q["split"], bool(q.get("gold"))) for q in questions)
    assert split_counts == Counter({("dev", True): 15, ("test", True): 5}), f"unexpected question split: {split_counts}"
    assert len(cases) == 24, f"expected 24 cases, got {len(cases)}"
    case_counts = Counter((c["split"], c["answerable"]) for c in cases)
    expected = Counter({("dev", True): 15, ("dev", False): 2, ("test", True): 5, ("test", False): 2})
    assert case_counts == expected, f"unexpected case split: {case_counts}"
    assert all("gold" not in case and "expected_answer" not in case for case in cases), "gold leaked into agent cases"

    for question in questions:
        assert question.get("expected_answer") and question.get("gold"), f"missing answer/evidence: {question['id']}"
        for evidence in question["gold"]:
            assert any(
                chunk["source_id"] == evidence["source_id"]
                and chunk["page"] == evidence["page"]
                and evidence["evidence"] in chunk["text"]
                for chunk in chunks
            ), f"gold evidence not found in source/page/chunks: {question['id']}"

    chunk_texts = [chunk["text"] for chunk in chunks]
    assert all(text in cache for text in chunk_texts), "a chunk embedding is missing from cache"
    vectors = [cache[text] for text in chunk_texts]
    dimensions = {len(vector) for vector in vectors}
    assert dimensions == {768}, f"expected 768-dimensional chunk embeddings, got {dimensions}"
    assert all(math.isfinite(float(value)) for vector in vectors for value in vector), "embedding has non-finite values"

    print(f"PASS: {len(chunks)} unique chunks; all {len(questions)} answerable golds resolve to source/page/evidence.")
    print("PASS: 24 agent cases (15+2 dev, 5+2 test); gold fields are excluded from agent input.")
    print(f"PASS: all chunk embeddings are cached, finite, and {next(iter(dimensions))}-dimensional.")


if __name__ == "__main__":
    main()
