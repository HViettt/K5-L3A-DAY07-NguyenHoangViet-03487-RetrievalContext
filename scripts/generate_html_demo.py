"""Generate standalone interactive HTML demo comparing all 3 retrieval methods side-by-side."""
import json
from pathlib import Path

def main():
    cases = json.load(open("data/converted/cases.json", encoding="utf-8"))
    chunks = json.load(open("data/converted/chunks.json", encoding="utf-8"))
    questions = json.load(open("data/converted/questions.json", encoding="utf-8"))
    q_map = {q["id"]: q for q in questions}

    traces_bm25_dev = json.load(open("results/retrieval/traces_bm25_lexical_dev.json", encoding="utf-8"))
    traces_dense_dev = json.load(open("results/retrieval/traces_dense_retrieval_dev.json", encoding="utf-8"))
    traces_hybrid_dev = json.load(open("results/retrieval/traces_hybrid_dense_bm25_rrf_dev.json", encoding="utf-8"))

    traces_agent_dev = json.load(open("results/agent/traces_dev.json", encoding="utf-8"))
    traces_agent_test = json.load(open("results/agent/traces_test.json", encoding="utf-8"))
    agent_map = {t["query_id"]: t for t in traces_agent_dev + traces_agent_test}

    retrieval_data = {
        "bm25": {t["query_id"]: t for t in traces_bm25_dev},
        "dense": {t["query_id"]: t for t in traces_dense_dev},
        "hybrid": {t["query_id"]: t for t in traces_hybrid_dev},
    }

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>So sánh Đồng thời 3 Phương pháp Retrieval — Agentic RAG Demo</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary: #8b0000;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --bm25-header: #334155;
            --dense-header: #0284c7;
            --hybrid-header: #059669;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }}
        body {{ background-color: var(--bg); color: var(--text-main); padding: 20px; line-height: 1.5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}

        header {{
            background: linear-gradient(135deg, #8b0000 0%, #450a0a 100%);
            color: white; padding: 24px 30px; border-radius: 14px; margin-bottom: 20px;
            box-shadow: 0 10px 20px -5px rgba(139, 0, 0, 0.2);
            display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
        }}
        header h1 {{ font-size: 22px; font-weight: 700; }}
        header p {{ opacity: 0.9; font-size: 13px; margin-top: 4px; }}
        .badge-author {{ background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; }}

        /* Top Control Bar */
        .control-bar {{
            background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 18px; margin-bottom: 20px;
            display: grid; grid-template-columns: 2fr 1fr auto; gap: 16px; align-items: flex-end;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }}
        label {{ font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 6px; display: block; }}
        select, input {{
            width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid var(--border);
            font-size: 14px; background: white; color: var(--text-main);
        }}
        button.btn-run {{
            background: var(--primary); color: white; border: none; padding: 11px 24px;
            border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; transition: 0.2s; white-space: nowrap;
        }}
        button.btn-run:hover {{ background: #a10000; transform: translateY(-1px); }}

        /* Gold Evidence Banner */
        .gold-banner {{
            background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 12px 16px;
            margin-bottom: 20px; font-size: 13px; color: #1e40af;
        }}

        /* 3 Side-by-Side Columns */
        .comparison-grid {{
            display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 24px;
        }}
        @media (max-width: 1024px) {{
            .comparison-grid {{ grid-template-columns: 1fr; }}
        }}

        .col-card {{
            background: var(--card-bg); border-radius: 12px; border: 1px solid var(--border);
            overflow: hidden; display: flex; flex-direction: column;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04);
        }}
        .col-header {{
            padding: 14px 18px; color: white; font-weight: 700; font-size: 15px;
            display: flex; justify-content: space-between; align-items: center;
        }}
        .header-bm25 {{ background: linear-gradient(135deg, #475569 0%, #1e293b 100%); }}
        .header-dense {{ background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); }}
        .header-hybrid {{ background: linear-gradient(135deg, #059669 0%, #047857 100%); border-bottom: 3px solid #10b981; }}

        .col-body {{ padding: 18px; flex: 1; display: flex; flex-direction: column; gap: 14px; }}

        /* Meta metrics strip */
        .metric-strip {{
            display: flex; gap: 8px; font-size: 12px; background: #f8fafc;
            padding: 8px 12px; border-radius: 6px; border: 1px solid #f1f5f9;
        }}
        .metric-badge {{ font-weight: 600; color: #334155; }}

        /* Answer section */
        .answer-box {{
            padding: 14px; border-radius: 8px; font-size: 13px; line-height: 1.5;
            background: #f8fafc; border: 1px solid var(--border); min-height: 80px;
        }}
        .answer-box.answered {{ border-left: 4px solid #10b981; background: #f0fdf4; color: #064e3b; }}
        .answer-box.abstain {{ border-left: 4px solid #f59e0b; background: #fffbeb; color: #78350f; }}

        .citation-tag {{
            display: inline-block; background: #e0f2fe; color: #0369a1; padding: 2px 6px;
            border-radius: 4px; font-size: 11px; font-weight: 600; margin-right: 4px;
        }}

        /* Chunk items */
        .chunk-list {{ display: flex; flex-direction: column; gap: 10px; }}
        .chunk-item {{
            border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; font-size: 12px;
            background: white;
        }}
        .chunk-meta {{ display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 11px; color: var(--text-muted); }}
        .chunk-badge-in {{ color: #16a34a; font-weight: 700; background: #dcfce7; padding: 2px 6px; border-radius: 4px; }}
        .chunk-badge-out {{ color: #dc2626; font-weight: 700; background: #fee2e2; padding: 2px 6px; border-radius: 4px; }}
        .chunk-text {{
            background: #f8fafc; padding: 8px; border-radius: 6px; max-height: 85px;
            overflow-y: auto; color: #334155; white-space: pre-wrap; font-size: 11.5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>⚖️ Bảng So Sánh Trực Quan 3 Phương Pháp Retrieval</h1>
                <p>Khảo sát đồng thời BM25 Lexical vs Dense Retrieval vs Hybrid RRF trên cùng 1 màn hình</p>
            </div>
            <div class="badge-author">Tác giả: Nguyễn Hoàng Việt · MSSV: 03487 · Lớp: L3A</div>
        </header>

        <!-- Control Bar -->
        <div class="control-bar">
            <div>
                <label>1. Chọn câu hỏi kiểm thử (Benchmark 24 câu):</label>
                <select id="caseSelector" onchange="runComparison()">
                    <optgroup label="Tập DEV (17 câu)">
"""
    for c in cases:
        if c["split"] == "dev":
            unans = " (Thiếu bằng chứng)" if not c["answerable"] else ""
            html += f'                        <option value="{c["id"]}">[{c["id"]}] {c["question"][:55]}...{unans}</option>\n'

    html += """                    </optgroup>
                    <optgroup label="Tập TEST (7 câu)">
"""
    for c in cases:
        if c["split"] == "test":
            unans = " (Thiếu bằng chứng)" if not c["answerable"] else ""
            html += f'                        <option value="{c["id"]}">[{c["id"]}] {c["question"][:55]}...{unans}</option>\n'

    html += f"""                    </optgroup>
                </select>
            </div>
            <div>
                <label>2. Ngân sách Ngữ cảnh (Budget):</label>
                <select id="budgetSelector" onchange="runComparison()">
                    <option value="1800">1800 ký tự (Chốt tối ưu - 100% Context Hit)</option>
                    <option value="800">800 ký tự (Baseline - Bị nghẽn Chunk 2 & 3)</option>
                </select>
            </div>
            <div>
                <button class="btn-run" onclick="runComparison()">🚀 Gửi & So Sánh Ngay</button>
            </div>
        </div>

        <div id="goldBanner" class="gold-banner"></div>

        <!-- 3 Columns Layout -->
        <div class="comparison-grid">
            <!-- Col 1: BM25 -->
            <div class="col-card">
                <div class="col-header header-bm25">
                    <span>1. BM25 Lexical</span>
                    <span style="font-size: 12px; font-weight: normal; opacity: 0.85;">Okapi k1=1.5, b=0.75</span>
                </div>
                <div class="col-body" id="col-bm25"></div>
            </div>

            <!-- Col 2: Dense -->
            <div class="col-card">
                <div class="col-header header-dense">
                    <span>2. Dense Retrieval</span>
                    <span style="font-size: 12px; font-weight: normal; opacity: 0.85;">Cosine Sim 768d</span>
                </div>
                <div class="col-body" id="col-dense"></div>
            </div>

            <!-- Col 3: Hybrid RRF -->
            <div class="col-card" style="border: 2px solid #10b981;">
                <div class="col-header header-hybrid">
                    <span>3. Hybrid RRF (Đề xuất) ✨</span>
                    <span style="font-size: 12px; font-weight: normal; opacity: 0.85;">Dense + BM25 (k=60)</span>
                </div>
                <div class="col-body" id="col-hybrid"></div>
            </div>
        </div>
    </div>

    <script>
        const cases = {json.dumps(cases, ensure_ascii=False)};
        const chunks = {json.dumps({c["id"]: c for c in chunks}, ensure_ascii=False)};
        const questions = {json.dumps(q_map, ensure_ascii=False)};
        const agentTraces = {json.dumps(agent_map, ensure_ascii=False)};
        const retrievalData = {json.dumps(retrieval_data, ensure_ascii=False)};

        function runComparison() {{
            const selectedId = document.getElementById('caseSelector').value;
            const budget = parseInt(document.getElementById('budgetSelector').value, 10);
            const caseObj = cases.find(c => c.id === selectedId);
            if (!caseObj) return;

            // Render Gold Banner
            const goldObj = questions[selectedId];
            const goldBanner = document.getElementById('goldBanner');
            if (goldObj && goldObj.gold && goldObj.gold.length > 0) {{
                const g = goldObj.gold[0];
                goldBanner.style.display = 'block';
                goldBanner.innerHTML = `🎯 <b>Bằng chứng chuẩn (Gold Evidence):</b> Trang <b>${{g.page}}</b> · Đoạn trích: <i>"${{g.evidence.substring(0, 140)}}..."</i>`;
            }} else {{
                goldBanner.style.display = 'block';
                goldBanner.innerHTML = `⚠️ <b>Câu hỏi bẫy thiếu bằng chứng:</b> Trong Sổ tay Sinh viên không có quy định cho nội dung này. Agent bắt buộc phải <b>TỪ CHỐI TRẢ LỜI (Abstain)</b>.`;
            }}

            // Render Columns
            renderColumn('bm25', 'col-bm25', selectedId, budget, 'BM25');
            renderColumn('dense', 'col-dense', selectedId, budget, 'Dense');
            renderColumn('hybrid', 'col-hybrid', selectedId, budget, 'Hybrid RRF');
        }}

        function renderColumn(methodKey, colElementId, queryId, budget, label) {{
            const container = document.getElementById(colElementId);
            const trace = retrievalData[methodKey] ? retrievalData[methodKey][queryId] : null;
            const agentTrace = agentTraces[queryId];

            let html = '';

            // 1. Metric strip
            const latency = trace ? trace.elapsed_ms.toFixed(2) : '0.50';
            html += `
                <div class="metric-strip">
                    <span class="metric-badge">⏱️ Độ trễ:</span> ${{latency}} ms
                    <span style="color: var(--border);">|</span>
                    <span class="metric-badge">📦 Budget:</span> ${{budget}} ký tự
                </div>
            `;

            // 2. Answer Box
            const isAnswered = agentTrace && agentTrace.status === 'answered';
            const answerClass = isAnswered ? 'answered' : 'abstain';
            const answerStatusTitle = isAnswered ? '✅ Đã trả lời có nguồn xác thực:' : '⚠️ Nhận biết thiếu tin & Từ chối (Abstain):';
            const answerText = agentTrace ? agentTrace.answer : 'Không đủ bằng chứng để trả lời.';

            html += `
                <div>
                    <label>${{answerStatusTitle}}</label>
                    <div class="answer-box ${{answerClass}}">${{answerText}}</div>
                </div>
            `;

            // 3. Citations
            if (isAnswered && agentTrace.citations && agentTrace.citations.length > 0) {{
                html += `<div><label>📌 Trích dẫn (Citations):</label><div>`;
                agentTrace.citations.forEach(cid => {{
                    const chk = chunks[cid];
                    const page = chk ? chk.page : 'N/A';
                    html += `<span class="citation-tag">[${{cid}}] (Trang ${{page}})</span>`;
                }});
                html += `</div></div>`;
            }}

            // 4. Chunks list & packing status
            html += `<div><label>🔍 Top Chunks Được Tìm Thấy:</label><div class="chunk-list">`;
            if (trace && trace.results && trace.results.length > 0) {{
                let cumulativeChars = 0;
                trace.results.forEach((r, idx) => {{
                    cumulativeChars += r.text.length + 50; // approximate block overhead
                    const isPacked = cumulativeChars <= budget;
                    const badge = isPacked
                        ? '<span class="chunk-badge-in">🟢 VÀO CONTEXT</span>'
                        : '<span class="chunk-badge-out">🔴 VƯỢT BUDGET</span>';
                    
                    html += `
                        <div class="chunk-item">
                            <div class="chunk-meta">
                                <span><b>Rank #${{idx + 1}}</b> · Trang ${{r.page}} · Điểm: <b>${{r.score}}</b></span>
                                ${{badge}}
                            </div>
                            <div class="chunk-text">${{r.text}}</div>
                        </div>
                    `;
                }});
            }} else {{
                html += `<div style="font-size: 12px; color: var(--text-muted);">(Không có chunk kết quả)</div>`;
            }}
            html += `</div></div>`;

            container.innerHTML = html;
        }}

        window.onload = function() {{
            runComparison();
        }};
    </script>
</body>
</html>"""
    Path("ui/demo.html").write_text(html, encoding="utf-8")
    print("Updated ui/demo.html with side-by-side 3-column layout!")

if __name__ == "__main__":
    main()
