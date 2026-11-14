# Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Chạy bài

Yêu cầu Python 3.11 trở lên. Từ thư mục gốc:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Điền `GEMINI_API_KEY` trong `.env` bằng khóa riêng của bạn. Không commit hoặc gửi file `.env`. Model mặc định và embedding được cấu hình trong `.env.example`.

```powershell
python -X utf8 -m unittest discover -s tests -v
python -X utf8 scripts/validate_submission.py
python -X utf8 src/run_experiments.py --retrieval-only
python -X utf8 src/run_experiments.py
```

Lệnh đầu chạy test offline. Lệnh retrieval dùng cache embedding 768 chiều và đo BM25, dense, hybrid RRF cùng hai thay đổi context budget. Lệnh cuối chạy cả retrieval và agent Gemini thật trên dev/test. Lỗi cấu hình hoặc API được ghi là lỗi; hệ thống không thay bằng câu trả lời heuristic.

`python -X utf8 src/run_experiments.py --offline-agent` chỉ tạo trace heuristic để debug trong `results/agent_offline/`; **không dùng kết quả này làm benchmark agent thật**.

Demo Streamlit tùy chọn: `streamlit run app.py`.

Nếu Gemini trả lỗi provider như 429/503, sau khi quota/dịch vụ sẵn sàng chạy `python -X utf8 src/run_experiments.py --retry-errors`. Lệnh này chỉ gọi lại query có `status: error`, giữ các trace đã thành công và giữ nhãn review đã hoàn tất; summary cũ sẽ bị xóa để tránh dùng nhầm số liệu trước khi tổng hợp lại.

Sau khi chạy agent, đọc từng trace và điền nhãn người chấm vào `results/agent/reviews_dev.json` và `reviews_test.json`. Chỉ sau khi review thủ công mới tổng hợp metrics:

```powershell
python -X utf8 starter/agent_metrics.py --cases data/converted/cases.json --traces results/agent/traces_dev.json --reviews results/agent/reviews_dev.json --split dev --output results/agent/summary_dev.json
python -X utf8 starter/agent_metrics.py --cases data/converted/cases.json --traces results/agent/traces_test.json --reviews results/agent/reviews_test.json --split test --output results/agent/summary_test.json
python -X utf8 starter/agent_metrics.py --cases data/converted/cases.json --traces results/agent/traces_test_baseline.json --reviews results/agent/reviews_test_baseline.json --split test --output results/agent/summary_test_baseline.json
```

Test mock chỉ kiểm tra code, không phải benchmark agent thật. Xem [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md), [quy tắc](RULES.md), [benchmark](BENCHMARK.md) và [cách nộp](SUBMISSION.md) trước khi nộp.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo cá nhân](report/individual/03487_NguyenHoangViet.md). `MSSV_HoTen.md` chỉ là mẫu. Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
