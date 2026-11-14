# Báo cáo Day 7

## Thông tin

- Họ và tên:
- MSSV:
- Lớp:
- Repository:

## 1. Dữ liệu và benchmark

- Nguồn, quyền truy cập/chia sẻ, số trang và số chunk:
- Vector store, embedding model, phiên bản và số chiều:
- Agent model và cách chạy:
- Benchmark dev/test, evidence và tiêu chí chấm:

## 2. Retrieval và thử nghiệm

So sánh ít nhất ba phương pháp retrieval trên cùng corpus và dev. Ghi cấu hình, filter, top-k, context budget, Hit@1, Evidence Hit@3, Context Evidence Rate, thời gian và chi phí.

| Cấu hình | Giả thuyết/thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian/chi phí |
|---|---|---|---:|---:|---:|---|

Phân tích ít nhất hai lỗi dev, hai cấu hình cải thiện, tiêu chí chọn và thời điểm chốt cấu hình trước khi chạy test.

## 3. Agent và đánh giá

Mô tả tool, điều kiện dừng, cách chọn context, citation và xử lý lỗi. Ghi nhãn thủ công từng câu có đáp án và câu thiếu evidence; không suy ra điểm đúng chỉ từ ID citation hoặc trang.

| Split | Tool use | Grounded answer 0/1/2 | Citation hợp lệ | Abstention đúng | Từ chối nhầm |
|---|---:|---|---:|---:|---:|

Liệt kê trace, review, summary, lệnh chạy, model, thời gian và chi phí.

## 4. Kết luận và giới hạn

Nêu kết quả có thể tái lập, giới hạn, lỗi còn lại và quyền truy cập dữ liệu.
