"""Implementation of 3 retrieval strategies: Dense, BM25, and Hybrid (RRF).

- DenseRetriever: Cosine similarity on 768-dim embeddings from Day 6 (gemini-embedding-001).
- BM25Retriever: Pure Python Okapi BM25 implementation (k1=1.5, b=0.75).
- HybridRetriever: Reciprocal Rank Fusion (RRF, k=60) combining Dense and BM25.
"""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def tokenize(text: str) -> list[str]:
    """Normalize and tokenize text for lexical retrieval."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"\w+", normalized)


def filter_chunk(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Check if a chunk matches the given metadata filters."""
    if not filters:
        return True
    meta = chunk.get("metadata", {})
    for k, v in filters.items():
        if k in chunk:
            if chunk[k] != v:
                return False
        elif k in meta:
            if meta[k] != v:
                return False
        else:
            return False
    return True


class DenseRetriever:
    """Dense retriever using Cosine Similarity on precomputed or live query vectors."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        cache_path: str | Path | None = None,
        embed_fn: Any | None = None,
    ) -> None:
        self.chunks = chunks
        self.embed_fn = embed_fn
        self.cache_path = Path(cache_path) if cache_path else Path("data/cache_embeddings.json")
        self._cache: dict[str, list[float]] = {}
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

        # Precompute chunk vectors
        self.chunk_ids = [c["id"] for c in chunks]
        self.vectors = np.array([self._cache.get(c["text"], [0.0] * 768) for c in chunks], dtype=np.float32)
        # Normalize vectors for fast cosine similarity via dot product
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.norm_vectors = self.vectors / norms

    def embed_query(self, query: str) -> np.ndarray:
        """Get query vector from custom embed_fn, cache, or live API."""
        if self.embed_fn is not None:
            vec = np.array(self.embed_fn(query), dtype=np.float32)
        elif query in self._cache:
            vec = np.array(self._cache[query], dtype=np.float32)
        else:
            # Fallback to Gemini API if network is available
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(f"Query embedding missing from cache and GEMINI_API_KEY not configured: {query[:30]}")
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            cfg = types.EmbedContentConfig(output_dimensionality=768)
            res = client.models.embed_content(
                model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
                contents=query,
                config=cfg,
            )
            vec = np.array(res.embeddings[0].values, dtype=np.float32)
            self._cache[query] = vec.tolist()
            if self.cache_path.exists():
                self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def search(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        """Return top_k most similar chunks using cosine similarity."""
        q_vec = self.embed_query(query)
        scores = np.dot(self.norm_vectors, q_vec)

        results = []
        for idx, chunk in enumerate(self.chunks):
            if filter_chunk(chunk, filters):
                results.append({
                    **chunk,
                    "score": round(float(scores[idx]), 4),
                    "strategy": "dense",
                })

        # Sort by score desc, id asc
        results.sort(key=lambda x: (-x["score"], x["id"]))
        return results[:top_k]


class BM25Retriever:
    """Okapi BM25 retriever (k1=1.5, b=0.75)."""

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 1.0

        # Term counts per doc and Document Frequency
        self.term_freqs = [Counter(toks) for toks in self.doc_tokens]
        self.df = Counter(term for tf in self.term_freqs for term in tf)
        self.N = len(chunks)
        # Compute BM25 IDF: ln((N - n + 0.5) / (n + 0.5) + 1)
        self.idf = {
            term: math.log((self.N - count + 0.5) / (count + 0.5) + 1.0)
            for term, count in self.df.items()
        }

    def search(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        """Return top_k chunks scored by Okapi BM25."""
        q_tokens = tokenize(query)
        q_freqs = Counter(q_tokens)
        results = []

        for idx, (chunk, tf, doc_len) in enumerate(zip(self.chunks, self.term_freqs, self.doc_lens)):
            if not filter_chunk(chunk, filters):
                continue
            score = 0.0
            for term, q_tf in q_freqs.items():
                if term in tf and term in self.idf:
                    f = tf[term]
                    denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    score += self.idf[term] * (f * (self.k1 + 1.0)) / denom
            if score > 0:
                results.append({
                    **chunk,
                    "score": round(float(score), 4),
                    "strategy": "bm25",
                })

        results.sort(key=lambda x: (-x["score"], x["id"]))
        return results[:top_k]


class HybridRetriever:
    """Hybrid Retriever combining Dense and BM25 via Reciprocal Rank Fusion (RRF)."""

    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever, rrf_k: int = 60) -> None:
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        """Run dense and bm25 retrievals and fuse results with RRF."""
        # Retrieve candidate pools
        dense_candidates = self.dense.search(query, filters=filters, top_k=min(30, len(self.dense.chunks)))
        bm25_candidates = self.bm25.search(query, filters=filters, top_k=min(30, len(self.bm25.chunks)))

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(dense_candidates, start=1):
            cid = item["id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, item in enumerate(bm25_candidates, start=1):
            cid = item["id"]
            if cid not in chunk_map:
                chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        fused = []
        for cid, score in rrf_scores.items():
            base_chunk = dict(chunk_map[cid])
            base_chunk["score"] = round(score, 6)
            base_chunk["strategy"] = "hybrid_rrf"
            fused.append(base_chunk)

        fused.sort(key=lambda x: (-x["score"], x["id"]))
        return fused[:top_k]


def pack_context(results: list[dict[str, Any]], budget: int = 800) -> tuple[str, list[str]]:
    """Pack chunks into context string within character budget."""
    blocks, ids, seen = [], [], set()
    for c in results:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        candidate_text = "\n\n".join(blocks + [block])
        if len(candidate_text) <= budget:
            blocks.append(block)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids
