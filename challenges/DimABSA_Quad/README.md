# Lời giải DimABSA_Quad - DimABSA Subtask 2: trích bộ tứ

Đề: [selfhost_kaggle/DimABSA_Quad](../../selfhost_kaggle/DimABSA_Quad/). Code: [solve.py](solve.py). Không dùng model pretrained.

## Kết quả (chấm bằng đúng metric của cuộc thi, `selfhost_kaggle/tools/cham.py`)

| | Public | Private |
|---|---|---|
| Baseline của đề | 0,305 | 0,281 |
| **Lời giải này** | 0,514 | 0,480 |

Thời gian chạy: 2,4 phút trên GTX 1650.

## Cách làm

1. Nhãn **BIO** cho aspect và opinion (`B-ASP, I-ASP, B-OPI, I-OPI, O`) từ bộ tứ train (khớp chuỗi không phân biệt hoa thường, trùng ranh giới token; bỏ NULL).
2. **BiLSTM 2 lớp + CNN ký tự**, word dropout 10%, 25 epoch, train từ đầu (dev cF1 0,63 khi tagger chỉ train trên train).
3. Ghép mỗi opinion với aspect gần nhất **cùng mệnh đề** (hoặc trong 8 token); không có aspect thì **bỏ** (test không có NULL, đúng nhận xét "nullDrop" của tài liệu AIO).
4. Category: LogisticRegression trên "aspect | opinion" (TF-IDF ký tự + từ). VA: Ridge trên opinion (1-3 gram) + mệnh đề.
5. Khử trùng lặp (aspect, category, opinion). Bản cuối train tagger trên train + dev.

## Ghi chú

Mốc tham khảo có pretrained (Gemma-3 12B QLoRA + self-consistency, tài liệu AIO): khoảng 0,64. Hướng tăng tiếp: ensemble 5 seed của tagger (bỏ phiếu theo token), hậu xử lý ranh giới opinion theo mẫu hay gặp ở train (giữ trạng từ và phủ định), category bằng mô hình nhìn cả câu.

## Chạy lại

```
python solve.py <thư_mục upload của đề>     # giải nén private_test.zip vào thư mục làm việc trước nếu muốn dự đoán cả private
python ../../selfhost_kaggle/tools/cham.py DimABSA_Quad submission.csv
```
