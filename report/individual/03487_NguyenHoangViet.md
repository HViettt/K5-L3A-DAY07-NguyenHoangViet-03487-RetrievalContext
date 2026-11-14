# Báo cáo Day 7 — Agentic RAG

- **Họ và tên:** Nguyễn Hoàng Việt
- **MSSV:** 03487
- **Lớp:** L3A
- **Repository:** `K5-L3A-DAY07-03487-NguyenHoangViet-RetrievalContext`

## 1. Dữ liệu và benchmark

Dùng *Sổ tay Sinh viên Đại học Bách khoa Hà Nội 2026*, lấy từ [Cổng Sổ tay sinh viên HUST](https://ctsv.hust.edu.vn/so-tay-sv). Bản PDF có 55 trang; pipeline dùng 53 trang nội dung và tạo 184 chunks. Tệp PDF không đưa vào Git; [data/SOURCE.md](../../data/SOURCE.md) ghi nguồn, mã băm và cách tải/tái tạo. Người chấm có thể tải được tài liệu từ nguồn công khai để chạy lại.

`data/cache_embeddings.json` chứa vector 768 chiều cho chunks và query, model `gemini-embedding-001`; dense cosine chạy trên cache cục bộ. Benchmark có 24 câu: 15 answerable và 2 unanswerable dev, 5 answerable và 2 unanswerable test. `scripts/validate_submission.py` xác nhận 184 ID duy nhất, toàn bộ 20 gold evidence khớp nguồn/trang/chunk, không rò rỉ gold sang input agent và vector chunk hợp lệ.

## 2. Retrieval và các thử nghiệm

So sánh BM25, dense retrieval và hybrid RRF trên cùng corpus, query dev và top-3. Đo thêm hai thay đổi context budget trên dev. Context budget tính theo ký tự và gồm ID/nguồn/trang. Bảng dưới đây là kết quả đã lưu trong `results/retrieval/`.

| Cấu hình | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median thời gian |
|---|---|---:|---:|---:|---:|
| BM25, budget 800 | dev | 12/15 (80.0%) | 15/15 (100%) | 5/15 (33.3%) | 0.63 ms |
| Dense, budget 800 | dev | 14/15 (93.3%) | 15/15 (100%) | 5/15 (33.3%) | 0.19 ms |
| Hybrid RRF, budget 800 | dev | 14/15 (93.3%) | 15/15 (100%) | 5/15 (33.3%) | 0.92 ms |
| Dense, budget 1600 | dev | 14/15 (93.3%) | 15/15 (100%) | 14/15 (93.3%) | 0.16 ms |
| **Hybrid RRF, budget 1800 (được chọn)** | dev | 14/15 (93.3%) | 15/15 (100%) | 15/15 (100%) | 0.94 ms |
| Hybrid RRF, budget 1800 | test | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 0.85 ms |
| BM25, budget 800 (baseline) | test | 5/5 (100%) | 5/5 (100%) | 0/5 (0%) | 0.55 ms |

Giả thuyết: tăng budget giúp giữ evidence trong context cuối dù ranking top-3 đã chứa evidence. Trên dev, dense budget 1600 nâng Context Evidence Rate từ 5/15 lên 14/15; hybrid budget 1800 đạt 15/15. BM25 dev có evidence đứng trong top-3 nhưng thường xếp thứ 2 hoặc 3, khiến budget 800 loại evidence ra khỏi context. Vì vậy chọn hybrid budget 1800 theo dev rồi mới chạy test. Baseline BM25 vẫn tìm được evidence trong top-3 test nhưng budget 800 không đưa evidence vào context. Audit từng câu trong `results/retrieval/reviews_context_test.json` chấm context của cấu hình hybrid 2/2 cho cả 5 câu; baseline 0/2 cho cả 5 câu. Agent nhiều lượt tìm vẫn có thể khôi phục câu trả lời, nên retrieval/context score và answer score được báo riêng. Trace truy xuất từng cấu hình, gồm context cuối, nằm trong `results/retrieval/traces_*.json`.

## 3. Agent, nhãn và kết quả

Agent dùng Gemini thật, tool retrieval, context có ID/nguồn/trang, citation, giới hạn lượt tìm và trace query/filter/candidates/context/answer/status/usage/latency. Lỗi provider không bị thay bằng câu trả lời heuristic. `gemini-2.5-flash` không khả dụng cho project trong lần thử (HTTP 404); các trace khởi đầu dùng `gemini-3.5-flash-lite`, sau khi chạm quota 429 đã chuyển sang `gemini-3.6-flash` và retry các case lỗi. Từng trace giữ tên model thực tế. Cả baseline BM25 và cấu hình hybrid được chọn đều chạy trên cùng 7 câu test.

| Run | Cases | Provider errors | Grounded answer (0/1/2) | Citation hỗ trợ | Câu có citation | Abstention đúng | False abstention |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dev, hybrid được chọn | 17 | 0 | 0/0/15 trên 15 câu answerable; mean 2.00/2 | 15/16 | 15/15 | 2/2 | 0/15 |
| Test, hybrid được chọn | 7 | 0 | 0/0/5 trên 5; mean 2.00/2 | 5/5 | 5/5 | 2/2 | 0/5 |
| Test, BM25 baseline | 7 | 0 | 0/0/5 trên 5; mean 2.00/2 | 5/5 | 5/5 | 2/2 | 0/5 |

Mỗi case đều gọi retrieval tool. Cả hai agent test trả lời đúng 5 câu có đáp án và từ chối 2 câu ngoài phạm vi tài liệu. Trên dev, q09 được context hỗ trợ đầy đủ bởi hai chunk cùng trang, nhưng câu trả lời chỉ dẫn citation tới chunk thứ nhất; citation không hỗ trợ mục thứ tư nên bị chấm không hợp lệ. Các citation còn lại được đối chiếu với nội dung chunk. Nhãn và lý do được lưu theo từng query trong `reviews_dev.json`, `reviews_test.json`, `reviews_test_baseline.json`.

**Minh bạch về việc chấm:** nhãn trong các file trên là audit của trợ lý dựa trên gold và chunk, không phải xác nhận của người chấm/giảng viên. Trước khi nộp, sinh viên nên đọc lại từng trace và xác nhận hoặc chỉnh các nhãn, đặc biệt q09. Script metrics chỉ tổng hợp nhãn, không tự chấm nội dung.

## 4. Lỗi dev và điều chỉnh

- **Evidence bị cắt khỏi context ở budget 800:** BM25, dense và hybrid đều có Evidence Hit@3 15/15 trên dev nhưng chỉ 5/15 context chứa evidence (33.3%). Giả thuyết là context packer loại chunk evidence do budget ký tự thấp. Tăng dense lên 1600 nâng lên 14/15 (93.3%); hybrid lên 1800 nâng lên 15/15 (100%). Giới hạn: đổi budget tăng lượng context, và thời gian đo trên cache cục bộ không tính latency dịch vụ embedding.
- **Citation q09 không bao phủ cả câu trả lời:** model nhận được hai chunk cần thiết trong context nhưng chỉ trích dẫn chunk đầu, khiến citation không hỗ trợ mục cuối. Answer vẫn đạt 2/2 nhờ hai chunk trong context; citation được chấm không hợp lệ. Có thể yêu cầu model trích dẫn mọi chunk hỗ trợ các phần của câu trả lời, nhưng chưa chạy lại để đo cải thiện vì quota; lỗi được giữ nguyên trong báo cáo.
- **Quota 429 trong lượt chạy đầu:** trace lưu lỗi provider thay vì giả câu trả lời; sau khi chuyển sang Gemini 3.6 Flash, giãn request và chỉ retry các case lỗi, trace dev/test hoàn chỉnh không còn provider error. Đây là khắc phục khả năng chạy, không phải thay đổi benchmark hay câu trả lời đã thành công.

## 5. Thời gian, token và chi phí

Tổng token theo usage trong trace: dev 19,379; hybrid test 9,909; BM25 baseline test 9,925. Median latency lần lượt 1.09 s, 11.88 s và 13.36 s; các lần retry/chờ quota làm một số latency test cao. API dollar cost chưa được đối chiếu từ billing project, nên không khẳng định chi phí bằng 0. Bảng quota cho thấy project chạm RPM của một số model; lần chạy hoàn chỉnh dùng Gemini 3.6 Flash sau khi đổi model và giới hạn khoảng cách request.

## 6. Chạy lại và trạng thái nộp

```powershell
python -X utf8 -m unittest discover -s tests -v
python -X utf8 scripts/validate_submission.py
python -X utf8 src/run_experiments.py --retrieval-only
python -X utf8 src/run_experiments.py --retry-errors
```

Lệnh agent cần `.env` cục bộ có `GEMINI_API_KEY`; không đưa `.env`/API key vào repo. Unit tests đạt 20/20; validator đạt; ba summary agent đều không còn provider error. Kết quả retrieval và agent có trace, review và summary trong `results/`. Thử thách nâng cao tài liệu từ 100 trang là bonus và chưa thực hiện.

**Trạng thái:** phần code, benchmark cơ bản, chạy agent trên toàn bộ dev/test, baseline test, trace và summary đã sẵn sàng để rà soát/nộp. Trước khi nộp, bạn cần xác nhận (1) nhãn review cuối cùng bằng cách đối chiếu trace theo rubric, và (2) quyền xem repo cùng khả năng truy cập/tái tạo PDF từ nguồn. Chi phí API chưa được tra từ billing project. Không commit secret. Mẫu `MSSV_HoTen.md` được giữ riêng làm template.
