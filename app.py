"""Interactive Streamlit Demo UI for Day 7 Agentic RAG System.

Compares all 3 retrieval methods (BM25, Dense, Hybrid RRF) simultaneously
on the same screen with side-by-side columns upon query submission.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

# Setup project path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import GeminiController, HeuristicController, run_agent
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, pack_context

# Page Configuration
st.set_page_config(
    page_title="So sánh 3 Phương pháp Retrieval — Agentic RAG Day 7",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_resources():
    chunks_path = PROJECT_ROOT / "data" / "converted" / "chunks.json"
    cases_path = PROJECT_ROOT / "data" / "converted" / "cases.json"
    questions_path = PROJECT_ROOT / "data" / "converted" / "questions.json"

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    q_map = {q["id"]: q for q in questions}

    dense = DenseRetriever(chunks)
    dense.strategy_name = "Dense (Cosine 768d)"

    bm25 = BM25Retriever(chunks)
    bm25.strategy_name = "BM25 Lexical"

    hybrid = HybridRetriever(dense, bm25)
    hybrid.strategy_name = "Hybrid (Dense + BM25 RRF)"

    return chunks, cases, q_map, dense, bm25, hybrid


chunks, cases, q_map, dense_retriever, bm25_retriever, hybrid_retriever = load_resources()

# Custom CSS for styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .col-header {
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        font-weight: 700;
        margin-bottom: 14px;
        color: white;
    }
    .header-bm25 { background: linear-gradient(135deg, #475569 0%, #1e293b 100%); }
    .header-dense { background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); }
    .header-hybrid { background: linear-gradient(135deg, #059669 0%, #047857 100%); border: 2px solid #34d399; }
    .answer-box {
        border-radius: 8px;
        padding: 12px;
        font-size: 0.95em;
        margin-bottom: 12px;
        min-height: 90px;
    }
    .badge-in { background-color: #dcfce7; color: #15803d; padding: 2px 8px; border-radius: 6px; font-weight: 600; font-size: 0.8em; }
    .badge-out { background-color: #fee2e2; color: #b91c1c; padding: 2px 8px; border-radius: 6px; font-weight: 600; font-size: 0.8em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("⚖️ Bảng So sánh Trực quan Đồng thời 3 Phương pháp Retrieval")
st.caption(
    "Đánh giá song song **BM25 Lexical** vs **Dense Retrieval** vs **Hybrid RRF** trên cùng một màn hình · "
    "**Tác giả:** Nguyễn Hoàng Việt (MSSV: 03487 - Lớp: L3A)"
)

# Sidebar
st.sidebar.header("⚙️ Cấu hình Thử nghiệm")

controller_choice = st.sidebar.radio(
    "Bộ điều phối (Controller Mode):",
    ["Heuristic (Ngoại tuyến, chính xác cao)", "Google Gemini API (gemini-2.0-flash)"],
    index=0,
    help="Chế độ Heuristic chạy trực tiếp nội bộ với độ trễ < 2ms và không tốn tiền API.",
)

top_k = st.sidebar.slider("Top-k Chunks:", min_value=1, max_value=5, value=3)
budget = st.sidebar.slider(
    "Ngân sách Ngữ cảnh (Context Budget):",
    min_value=400,
    max_value=3000,
    value=1800,
    step=100,
    help="Chỉnh về 800 để quan sát lỗi nghẽn context; chỉnh lên 1800 để đạt 100% bằng chứng.",
)

filter_category = st.sidebar.selectbox(
    "Lọc theo Phân mục (Category Filter):",
    ["Tất cả (Không lọc)", "quy_che_quy_dinh", "diem_ren_luyen", "hoc_bong_an_sinh", "gioi_thieu_vien_khoa"],
    index=0,
)
selected_filter = None if filter_category.startswith("Tất cả") else {"category": filter_category}

# Benchmark Question Selector
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Chọn nhanh từ Benchmark (24 câu)")
benchmark_options = ["(Tự nhập câu hỏi của bạn...)"] + [
    f"[{c['id']}] [{c['split'].upper()}] {c['question']} {'(Thiếu tin)' if not c['answerable'] else ''}"
    for c in cases
]
selected_option = st.sidebar.selectbox("Câu hỏi Benchmark:", benchmark_options, index=1)

# Main Query Input
default_question = ""
current_gold = None
if selected_option != "(Tự nhập câu hỏi của bạn...)":
    qid = selected_option.split("]")[0][1:]
    case_obj = next((c for c in cases if c["id"] == qid), None)
    if case_obj:
        default_question = case_obj["question"]
    if qid in q_map:
        current_gold = q_map[qid]["gold"]

question_input = st.text_input("Nhập câu hỏi cần tra cứu:", value=default_question)
run_clicked = st.button("🚀 Gửi Câu Hỏi & So Sánh Đồng Thời 3 Phương Pháp", type="primary")

if run_clicked or default_question:
    if not question_input.strip():
        st.warning("Vui lòng nhập câu hỏi.")
    else:
        # Determine Controller
        if controller_choice.startswith("Google Gemini"):
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE" or api_key.startswith("YOUR_"):
                st.error("Chưa có API key trong .env, chuyển về chế độ Heuristic.")
                controller = HeuristicController(chunks)
            else:
                try:
                    controller = GeminiController(api_key=api_key)
                except Exception:
                    controller = HeuristicController(chunks)
        else:
            controller = HeuristicController(chunks)

        # Helper function to check evidence hit
        def check_evidence_hit(c_list, gold_list):
            if not gold_list:
                return False
            for c in c_list:
                for g in gold_list:
                    if c["page"] == g["page"] and g["evidence"] in c["text"]:
                        return True
            return False

        # Run 3 methods simultaneously
        methods = [
            ("BM25 Lexical", bm25_retriever, "header-bm25"),
            ("Dense (Cosine 768d)", dense_retriever, "header-dense"),
            ("Hybrid RRF (Dense + BM25) ✨", hybrid_retriever, "header-hybrid"),
        ]

        results = []
        for name, ret, header_cls in methods:
            t0 = time.perf_counter()
            agent_res = run_agent(
                question=question_input,
                controller=controller,
                retriever=ret,
                filters=selected_filter,
                max_searches=3,
                top_k=top_k,
                budget=budget,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Find search event
            search_ev = next((e for e in agent_res["events"] if e.get("event") == "search"), None)
            res_chunks = search_ev.get("results", []) if search_ev else []
            packed_ids = set(search_ev.get("context_ids", [])) if search_ev else set()
            packed_chunks = [c for c in res_chunks if c["id"] in packed_ids]

            # Evaluation if gold is available
            hit1 = False
            hit3 = False
            context_hit = False
            if current_gold and res_chunks:
                hit1 = check_evidence_hit([res_chunks[0]], current_gold)
                hit3 = check_evidence_hit(res_chunks[:3], current_gold)
                context_hit = check_evidence_hit(packed_chunks, current_gold)

            results.append({
                "name": name,
                "header_cls": header_cls,
                "status": agent_res["status"],
                "answer": agent_res["answer"],
                "citations": agent_res["citations"],
                "events": agent_res["events"],
                "elapsed_ms": elapsed_ms,
                "res_chunks": res_chunks,
                "packed_ids": packed_ids,
                "packed_count": len(packed_ids),
                "hit1": hit1,
                "hit3": hit3,
                "context_hit": context_hit,
            })

        st.markdown("---")

        # 1. SUMMARY COMPARISON BANNER
        st.subheader("📊 Bảng Đối Soát Hiệu Năng Tổng Hợp Trực Quan")
        comp_cols = st.columns(4)
        comp_cols[0].metric("Thời gian xử lý (Latency)", f"{results[2]['elapsed_ms']:.1f} ms", f"{results[2]['elapsed_ms'] - results[0]['elapsed_ms']:+.1f} ms vs BM25")
        comp_cols[1].metric("Top-1 Score (Hybrid RRF)", f"{results[2]['res_chunks'][0]['score'] if results[2]['res_chunks'] else 'N/A'}")
        comp_cols[2].metric("Số Chunk vào Context", f"{results[2]['packed_count']} / {top_k} chunks", f"Budget: {budget} chars")
        comp_cols[3].metric("Trạng thái Phản hồi", "✅ Đã trả lời" if results[2]["status"] == "answered" else "⚠️ Từ chối (Abstain)")

        if current_gold:
            st.info(f"🎯 **Bằng chứng chuẩn (Gold Ground Truth):** Trang `{current_gold[0]['page']}` — Đoạn trích: *\"{current_gold[0]['evidence'][:120]}...\"*")

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. THREE SIDE-BY-SIDE COLUMNS
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        for col, r in zip(cols, results):
            with col:
                st.markdown(f'<div class="col-header {r["header_cls"]}">{r["name"]}</div>', unsafe_allow_html=True)

                # Status & Latency Badges
                st.write(f"⏱️ **Độ trễ:** `{r['elapsed_ms']:.1f} ms` | 📦 **Nạp Context:** `{r['packed_count']}/{len(r['res_chunks'])}` chunks")

                # Evidence Match Badge (if available)
                if current_gold:
                    hit_color = "🟢" if r["context_hit"] else ("🟡" if r["hit3"] else "🔴")
                    hit_text = "Có Evidence trong Context" if r["context_hit"] else ("Có trong Top-3 nhưng BỊ NGHẼN BUDGET" if r["hit3"] else "Không tìm thấy Evidence")
                    st.write(f"{hit_color} **Đánh giá:** {hit_text}")

                # Answer Box
                st.markdown("##### 💬 Phản hồi của Agent:")
                box_bg = "#ecfdf5" if r["status"] == "answered" else "#fffbeb"
                box_border = "#10b981" if r["status"] == "answered" else "#f59e0b"
                st.markdown(
                    f"""
                    <div class="answer-box" style="background-color: {box_bg}; border: 1px solid {box_border};">
                        {r['answer']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Citations
                st.markdown("##### 📌 Nguồn Trích Dẫn (Citations):")
                if r["citations"]:
                    for cid in r["citations"]:
                        chk = next((c for c in chunks if c["id"] == cid), None)
                        page_num = chk["page"] if chk else "N/A"
                        cat = chk["metadata"].get("category") if chk else "N/A"
                        st.markdown(f"- `[{cid}]` (Trang {page_num} · {cat})")
                else:
                    st.caption("*(Không có trích dẫn / Đã từ chối)*")

                # Chunks Ranking
                st.markdown("##### 🔍 Danh Sách Chunks Được Tìm Thấy:")
                for rank_idx, chk in enumerate(r["res_chunks"], start=1):
                    is_in = chk["id"] in r["packed_ids"]
                    badge_html = '<span class="badge-in">🟢 VÀO CONTEXT</span>' if is_in else '<span class="badge-out">🔴 VƯỢT BUDGET</span>'
                    exp_label = f"#{rank_idx} | Trang {chk['page']} | Điểm: {chk['score']} | {chk['id'][-12:]}"
                    with st.expander(exp_label, expanded=(rank_idx == 1)):
                        st.markdown(f"**Trạng thái:** {badge_html}", unsafe_allow_html=True)
                        st.markdown(f"**Độ dài:** {len(chk['text'])} ký tự | **Phân loại:** `{chk['metadata'].get('category')}`")
                        st.text_area("Nội dung:", value=chk["text"], height=100, key=f"{r['name']}_{rank_idx}")
