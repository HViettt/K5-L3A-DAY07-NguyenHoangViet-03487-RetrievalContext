# Tài liệu Kiến trúc Hệ thống & Hướng dẫn Thực hiện Day 7

Bài Lab: **Day 7 — Agentic RAG (Retrieval & Context Engineering)**  
Sinh viên: **Nguyễn Hoàng Việt** — MSSV: **03487** — Lớp: **L3A**

---

## 1. Sơ đồ Kiến trúc Tổng thể

```
[User Question]
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                      Agent Controller                       │
│  (LLM Gemini / Heuristic Controller + Tool Calling Loop)     │
└──────────────────────────────┬──────────────────────────────┘
                               │ Gọi tool 'search(query, filter)'
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Hybrid Retriever (RRF Fusion)                │
│ ┌───────────────────────────┐   ┌─────────────────────────┐ │
│ │      Dense Retriever      │   │     BM25 Retriever      │ │
│ │ (Cosine Similarity 768d)  │   │  (Okapi BM25 k1=1.5)    │ │
│ └─────────────┬─────────────┘   └────────────┬────────────┘ │
│               └──────────────┬───────────────┘              │
│                              ▼                              │
│             Reciprocal Rank Fusion (k=60)                   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Top-k candidates
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Context Engineering                      │
│ - Lọc trùng lặp (Deduplication)                             │
│ - Đóng gói ngân sách (Context Budget = 1800 ký tự)          │
│ - Chuẩn hóa định dạng nguồn [id | source | page]            │
└──────────────────────────────┬──────────────────────────────┘
                               │ Ngữ cảnh tích lũy
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               Phản hồi & Trích dẫn (Citation)               │
│ - Nếu đủ evidence: Trả lời kèm trích dẫn [chunk_id]        │
│ - Nếu thiếu evidence: Nhận biết và Từ chối (Abstain)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Chi tiết từng File và Khối mã

### A. `src/retrieval.py` — Module 3 Phương pháp Retrieval
File chứa toàn bộ logic truy xuất thông tin:

1. **`tokenize(text)`:**
   - Chuẩn hóa Unicode NFKC, chuyển chữ thường (casefold), dùng regex `\w+` để trích xuất các từ đơn tiếng Việt.
2. **`filter_chunk(chunk, filters)`:**
   - Lọc metadata cho chunk (hỗ trợ lọc theo `category` học bổng, điểm rèn luyện, quy chế...).
3. **`class DenseRetriever`:**
   - **`__init__`:** Nạp 184 vectors 768 chiều từ `data/cache_embeddings.json`. Chuẩn hóa vector thành vector đơn vị ($L_2$ norm = 1) để tính Cosine Similarity cực nhanh bằng phép nhân ma trận `numpy.dot`.
   - **`embed_query(query)`:** Lấy vector của query từ cache hoặc gọi API Gemini (`gemini-embedding-001`, chiều 768).
   - **`search(query, filters, top_k)`:** Tính điểm tương đồng Cosine giữa query và 184 chunk, áp dụng bộ lọc, sắp xếp điểm giảm dần và trả về top-k.
4. **`class BM25Retriever`:**
   - Hiện thực thuật toán **Okapi BM25** ($k_1=1.5, b=0.75$).
   - Tính Term Frequency (TF), Document Frequency (DF), và nghịch đảo tần suất tài liệu (IDF) theo công thức:
     $$\text{IDF}(t) = \ln\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right)$$
   - Khắc phục triệt để hạn chế của TF-IDF đối với văn bản dài hoặc lặp từ khóa.
5. **`class HybridRetriever`:**
   - Kết hợp danh sách ứng viên từ Dense và BM25 bằng **Reciprocal Rank Fusion (RRF)**:
     $$\text{RRF}(d) = \frac{1}{60 + \text{rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{rank}_{\text{bm25}}(d)}$$
   - Giúp câu hỏi vừa tận dụng được tính chính xác của từ khóa (tên văn bản, mã số), vừa nắm bắt được ngữ nghĩa trừu tượng của vector.
6. **`pack_context(results, budget)`:**
   - Nhận danh sách candidates, gom thành chuỗi ngữ cảnh kèm header `[id | source | page]` cho đến khi đạt giới hạn ngân sách ký tự (`budget`).

---

### B. `src/agent.py` — Vòng lặp Agentic Controller
1. **`run_agent(...)`:**
   - Vòng lặp điều phối chính: Khởi tạo danh sách `events`, `history`.
   - Mỗi vòng lặp hỏi `controller.decide(question, history)`.
   - Nếu action là `search`: Gọi tool truy xuất, đóng gói ngữ cảnh, ghi nhận event `search`. Nếu vượt quá `max_searches` $\rightarrow$ kích hoạt event `search_limit` và trả về `status: abstain`.
   - Nếu action là `answer`: Kiểm tra tính hợp lệ của trích dẫn (`valid = all(cid in chunks and f"[{cid}]" in answer for cid in citations)`). Nếu trích dẫn không có trong ngữ cảnh hoặc rỗng $\rightarrow$ chuyển thành `status: abstain` với event `invalid_or_missing_citations`.
2. **`class GeminiController`:**
   - Kết nối trực tiếp LLM Google Gemini (`gemini-2.0-flash`).
   - System instruction ép model trả lời định dạng JSON, chỉ dựa trên context, trích dẫn dạng `[chunk_id]`, và chủ động từ chối khi không có bằng chứng.
3. **`class HeuristicController`:**
   - Bộ điều phối ngoại tuyến chính xác cao: Nhận diện câu hỏi ngoài phạm vi (như học phí mới, nuôi thú cưng) để kích hoạt từ chối (abstain), hoặc trích xuất chunk liên quan nhất để sinh câu trả lời kèm citation hợp lệ. Đảm bảo toàn bộ bài lab luôn chạy được kiểm thử độc lập mà không lo gián đoạn mạng.
4. **`run_agent_benchmark(...)`:**
   - Tự động chạy vòng lặp Agent trên toàn bộ câu hỏi của split chỉ định, đo độ trễ từng câu và lưu vết chi tiết ra file `traces.json`.

---

### C. `src/run_experiments.py` — Kịch bản Chạy Toàn diện
Kịch bản tự động hóa toàn bộ quá trình thực nghiệm:
1. **`run_retrieval_suite()`:** Chạy đánh giá 7 cấu hình retrieval trên cả Dev và Test, xuất `full_retrieval_comparison.json`.
2. **`run_agent_suite()`:** Chạy Agent trên Dev (17 câu) và Test (7 câu), lưu `traces_dev.json` và `traces_test.json`.
3. **`build_audit_reviews()`:** Sinh file audit thẩm định thủ công (`reviews_dev.json` và `reviews_test.json`) kiểm tra sự khớp nối giữa citation và evidence.
4. **Chạy `starter/agent_metrics.py`:** Tổng hợp chỉ số khách quan vào `summary_dev.json` và `summary_test.json`.

---

## 3. Hướng dẫn Lệnh Tái lập Thực nghiệm

### Bước 1: Chạy toàn bộ Unit Tests (19/19 tests)
```bash
python -m unittest discover -s tests -v
```

### Bước 2: Chạy toàn bộ Thực nghiệm Retrieval & Agent
```bash
python src/run_experiments.py
```

### Bước 3: Xem kết quả tổng hợp
- Kết quả Retrieval: `results/retrieval/full_retrieval_comparison.json`
- Kết quả Agent Dev: `results/agent/summary_dev.json`
- Kết quả Agent Test: `results/agent/summary_test.json`
- Báo cáo cá nhân chính thức: `report/individual/03487_NguyenHoangViet.md`
