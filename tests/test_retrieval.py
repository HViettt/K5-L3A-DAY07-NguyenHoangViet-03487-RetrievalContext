"""Unit tests for src/retrieval.py (Dense, BM25, Hybrid RRF)."""
import unittest
from src.retrieval import DenseRetriever, BM25Retriever, HybridRetriever, pack_context, tokenize

SAMPLE_CHUNKS = [
    {
        "id": "c1",
        "text": "Trường Đại học Bách khoa Hà Nội thành lập năm 1956.",
        "source_id": "handbook",
        "source": "handbook.pdf",
        "page": 2,
        "metadata": {"category": "lich_su"}
    },
    {
        "id": "c2",
        "text": "Quy chế tính điểm rèn luyện sinh viên theo tiêu chí.",
        "source_id": "handbook",
        "source": "handbook.pdf",
        "page": 10,
        "metadata": {"category": "diem_ren_luyen"}
    },
    {
        "id": "c3",
        "text": "Chính sách học bổng Trần Đại Nghĩa hỗ trợ sinh viên khó khăn.",
        "source_id": "handbook",
        "source": "handbook.pdf",
        "page": 17,
        "metadata": {"category": "hoc_bong"}
    }
]

class RetrievalTests(unittest.TestCase):
    def test_tokenize(self):
        toks = tokenize("Đại học Bách Khoa Hà Nội! Năm 1956.")
        self.assertIn("đại", toks)
        self.assertIn("học", toks)
        self.assertIn("bách", toks)
        self.assertIn("1956", toks)

    def test_bm25_search_and_filter(self):
        bm25 = BM25Retriever(SAMPLE_CHUNKS)
        results = bm25.search("năm thành lập trường 1956", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], "c1")
        self.assertEqual(results[0]["strategy"], "bm25")

        # Test filter
        filtered = bm25.search("sinh viên", filters={"category": "hoc_bong"}, top_k=2)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["id"], "c3")

    def test_hybrid_rrf_combines_ranks(self):
        bm25 = BM25Retriever(SAMPLE_CHUNKS)
        dense = DenseRetriever(SAMPLE_CHUNKS, embed_fn=lambda q: [0.1] * 768)
        hybrid = HybridRetriever(dense, bm25)
        results = hybrid.search("học bổng Trần Đại Nghĩa", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["strategy"], "hybrid_rrf")

    def test_pack_context_budget(self):
        packed, ids = pack_context(SAMPLE_CHUNKS, budget=120)
        self.assertIn("c1", ids)
        self.assertLessEqual(len(packed), 120)

if __name__ == "__main__":
    unittest.main()
