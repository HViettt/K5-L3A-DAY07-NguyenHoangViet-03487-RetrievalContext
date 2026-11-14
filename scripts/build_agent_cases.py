"""Build agent_cases.json with 24 benchmark questions (17 dev: 15 ans + 2 unans; 7 test: 5 ans + 2 unans)."""
import json

def main():
    questions = json.load(open('data/converted/questions.json', encoding='utf-8'))
    
    cases = []
    # 20 answerable questions from Day 6
    for q in questions:
        cases.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer"
        })
    
    # 4 unanswerable questions (2 dev, 2 test)
    unanswerable = [
        {
            "id": "q_unans_dev_01",
            "split": "dev",
            "question": "Mức học phí của chương trình tiên tiến năm học 2026-2027 là bao nhiêu triệu đồng một tín chỉ?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Trong sổ tay sinh viên 2026 chỉ có thông tin giới thiệu chung về chương trình tích hợp và học bổng, không có bảng biểu mức học phí cụ thể theo tín chỉ cho năm học 2026-2027."
        },
        {
            "id": "q_unans_dev_02",
            "split": "dev",
            "question": "Sinh viên có được phép nuôi hoặc mang vật nuôi (chó, mèo) vào trong phòng ký túc xá không?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Trong sổ tay sinh viên 2026 không có điều khoản quy chế chi tiết liên quan đến việc nuôi hoặc mang thú cưng/vật nuôi vào ký túc xá."
        },
        {
            "id": "q_unans_test_01",
            "split": "test",
            "question": "Điều kiện và thủ tục để sinh viên được cấp học bổng du học toàn phần tại Đại học Tokyo Nhật Bản là gì?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Trong sổ tay sinh viên chỉ đề cập đến Học bổng Trần Đại Nghĩa và khuyến khích học tập nội bộ, không có thông tin học bổng du học Đại học Tokyo Nhật Bản."
        },
        {
            "id": "q_unans_test_02",
            "split": "test",
            "question": "Quy trình thủ tục đăng ký xin cấp lại biển số xe máy bị mất tại nhà gửi xe trường Bách khoa gồm những bước nào?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Trong sổ tay sinh viên không có nội dung hướng dẫn thủ tục giải quyết việc mất hay cấp lại biển số xe máy tại bãi đỗ xe."
        }
    ]
    
    cases.extend(unanswerable)
    
    dev_cases = [c for c in cases if c['split'] == 'dev']
    test_cases = [c for c in cases if c['split'] == 'test']
    print(f"Total benchmark cases: {len(cases)}")
    print(f"Dev: {len(dev_cases)} (Answerable: {sum(c['answerable'] for c in dev_cases)}, Unanswerable: {sum(not c['answerable'] for c in dev_cases)})")
    print(f"Test: {len(test_cases)} (Answerable: {sum(c['answerable'] for c in test_cases)}, Unanswerable: {sum(not c['answerable'] for c in test_cases)})")
    
    with open('data/converted/cases.json', 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print("Saved to data/converted/cases.json")

if __name__ == '__main__':
    main()
