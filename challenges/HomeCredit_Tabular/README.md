# Lời giải HomeCredit_Tabular - Dự đoán vỡ nợ (dữ liệu bảng nhiều file)

Đề: [selfhost_kaggle/HomeCredit_Tabular](../../selfhost_kaggle/HomeCredit_Tabular/). Code: [solve.py](solve.py). Không dùng model pretrained.

## Kết quả (chấm bằng đúng metric của cuộc thi, `selfhost_kaggle/tools/cham.py`)

| | Public | Private |
|---|---|---|
| Baseline của đề | 0,761 | 0,747 (chỉ bảng chính) |
| **Lời giải này** | 0,779 | 0,771 |

Thời gian chạy: 3,3 phút, CPU.

## Cách làm

1. **Gộp `bureau.csv` theo id**: số khoản vay, tỉ lệ khoản đang nợ, tỉ lệ / số ngày quá hạn, tổng tiền vay và dư nợ, tỉ lệ dư nợ / tiền vay, số lần gia hạn, ngày vay gần nhất.
2. **Gộp `previous_application.csv`**: số hồ sơ, tỉ lệ được duyệt / bị từ chối, tiền vay trung bình và lớn nhất, tỉ lệ tiền được cấp / tiền xin, kỳ hạn, hồ sơ gần nhất có bị từ chối không. Giá trị `365243` trong cột `DAYS_*` đổi thành thiếu.
3. **Đặc trưng tỉ lệ**: khoản vay / thu nhập, tiền trả hằng kỳ / thu nhập, khoản vay / giá hàng, kỳ hạn, số năm làm việc / tuổi, trung bình / nhỏ nhất / tích của `EXT_SOURCE_1..3`, dư nợ bureau / thu nhập.
4. `HistGradientBoostingClassifier` (chỉ cần sklearn) có early stopping; val theo thời gian (04-06/2020): 0,768. Bản cuối train toàn bộ, trung bình 3 seed.

## Ghi chú

Gộp bảng phụ cho phần lớn mức tăng. Muốn tăng tiếp: thêm `installments_payments`, `credit_card_balance` (bản tự host chưa đưa vào), gộp theo cửa sổ thời gian (12 / 24 tháng gần nhất), LightGBM + CatBoost và blend theo hạng.

## Chạy lại

```
python solve.py <thư_mục upload của đề>     # giải nén private_test.zip vào thư mục làm việc trước nếu muốn dự đoán cả private
python ../../selfhost_kaggle/tools/cham.py HomeCredit_Tabular submission.csv
```
