# ViQuAD_ExtractiveQA - thử nghiệm (CHƯA vượt baseline rõ ràng)

Đề: [selfhost_kaggle/ViQuAD_ExtractiveQA](../../selfhost_kaggle/ViQuAD_ExtractiveQA/). Code: [solve.py](solve.py). Không dùng model pretrained.

Đây **không phải** lời giải tốt. Mình giữ lại vì kết quả và phần chẩn đoán cho thấy dạng bài này khó thế nào khi không có pretrained, và nên đầu tư vào đâu.

## Kết quả

| | Dev | Public | Private |
|---|---|---|---|
| Để trống mọi câu | 0,333 | 0,274 | khoảng 0,33 |
| Baseline của đề (trả cả câu khi trùng ≥ 80% từ câu hỏi) | 0,337 | 0,276 | 0,337 |
| **Thử nghiệm này** | 0,342 | 0,299 | 0,336 |

Thời gian: 6 phút, CPU.

## Cách làm đã thử

Giống bài luyện [d03_hoi_dap](../../luyen_tap/CV/d03_hoi_dap/) (dữ liệu giả lập ở đó đạt 0,88):

1. Chọn 2 câu có tổng IDF của từ câu hỏi lớn nhất.
2. Sinh span 1-6 âm tiết.
3. Đặc trưng span: độ dài, chữ số, chữ hoa, khoảng cách tới từ câu hỏi, IDF trong cửa sổ, từ trái / phải, loại câu hỏi và tương tác.
4. XGBoost hồi quy F1 của span (train trên 3.000 câu hỏi, khoảng 1,1 triệu span).
5. Ngưỡng để trống chọn trên dev.

## Chẩn đoán (trên 400 câu hỏi dev có đáp án)

- **Trần của bộ sinh ứng viên**: F1 tốt nhất đạt được trong các span là 0,77; 48,5% câu có span trùng khớp hoàn toàn. Bước chọn câu không phải nút thắt chính.
- **Đáp án dài**: trung vị 6 âm tiết, chỉ 54,5% dài ≤ 6. Giới hạn 6 âm tiết cắt mất gần nửa đáp án; nên để 10-15 (VIQA: 92% đáp án ≤ 10 âm tiết, dữ liệu này dài hơn).
- **Xếp hạng**: với hàng trăm span mỗi câu hỏi, đặc trưng thô không đủ chọn đúng span; mô hình cho điểm thấp đồng đều nên ngưỡng tối ưu (0,30) khiến gần như mọi câu để trống.

## Hướng nên làm (theo thứ tự)

1. Span tới 12-15 âm tiết, nhưng **chỉ** sinh span bắt đầu / kết thúc ở ranh giới cụm hợp lý (sau dấu phẩy, sau "là", "của", "vào", "năm"...).
2. Đủ bộ đặc trưng của [challenges/VietnameseExtract](../VietnameseExtract/) (37 đặc trưng) và `XGBRanker` nhóm theo câu hỏi thay cho hồi quy.
3. **Tách quyết định để trống**: một bộ phân loại riêng "câu hỏi có đáp án không" (đặc trưng: điểm câu tốt nhất, tỉ lệ từ câu hỏi khớp, chênh lệch điểm span tốt nhất và nhì, loại câu hỏi), rồi mới chọn span.
4. Mạng đọc hiểu nhỏ train từ đầu (BiLSTM + attention giữa câu hỏi và đoạn văn, kiểu BiDAF) trên 25.000 câu hỏi train; có đủ dữ liệu để thử.

## Chạy lại

```
python solve.py <thư_mục upload của đề>     # giải nén private_test.zip vào thư mục làm việc trước nếu muốn dự đoán cả private
python ../../selfhost_kaggle/tools/cham.py ViQuAD_ExtractiveQA submission.csv
```
