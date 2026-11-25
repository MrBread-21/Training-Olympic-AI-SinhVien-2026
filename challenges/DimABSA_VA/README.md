# Lời giải DimABSA_VA - DimABSA Subtask 1: hồi quy Valence-Arousal

Đề: [selfhost_kaggle/DimABSA_VA](../../selfhost_kaggle/DimABSA_VA/). Code: [solve.py](solve.py). Không dùng model pretrained.

## Kết quả (chấm bằng đúng metric của cuộc thi, `selfhost_kaggle/tools/cham.py`)

| | Public | Private |
|---|---|---|
| Baseline của đề | 1,901 | 1,896 |
| **Lời giải này** | 1,863 | 1,871 |

Thời gian chạy: 5 giây, CPU.

## Cách làm

1. Tách từ chung cho train (đã tách sẵn, chữ thường: `ca n ' t`) và test (câu gốc: `can't`): chữ thường, gộp phủ định về `not`, tách `'s` / `'re`...
2. Ba góc nhìn cho mỗi khía cạnh (khía cạnh thay bằng `ASP`): **mệnh đề chứa khía cạnh** (cắt tại dấu câu và but / and / though / while...), cửa sổ ±8 từ, cả câu; TF-IDF từ 1-3 gram + ký tự cho mệnh đề, trọng số giảm dần cho cửa sổ và cả câu.
3. Ridge riêng cho valence và arousal; arousal trộn thêm dự đoán từ |V - 5| (quan hệ chữ V của dữ liệu).
4. Giãn dự đoán bằng hồi quy tuyến tính học trên dev (mô hình hồi quy luôn co về trung bình), kẹp [1, 9], làm tròn 2 chữ số. Dev RMSE_VA 1,35.

## Ghi chú

**Giới hạn**: chỉ hơn baseline một chút. Trên dev (cùng nguồn với test) được 1,35 nhưng test khó hơn hẳn: các khía cạnh tiêu cực (V thật khoảng 2,4) vẫn bị đoán khoảng 5, vì train lệch mạnh về tích cực và câu tiêu cực trong test dùng từ ngữ ít gặp ở train. Thử trọng số mẫu theo tần suất mức valence: không giúp. Mốc tham khảo có pretrained (RoBERTa-large, tài liệu AIO): 1,10. Hướng nên thử: từ điển cảm xúc học từ train (valence trung bình của mỗi từ) làm đặc trưng cho cây quyết định, BiGRU từ đầu có đánh dấu vị trí khía cạnh.

## Chạy lại

```
python solve.py <thư_mục upload của đề>     # giải nén private_test.zip vào thư mục làm việc trước nếu muốn dự đoán cả private
python ../../selfhost_kaggle/tools/cham.py DimABSA_VA submission.csv
```
