# HomeCredit_Tabular - nội dung từng ô trên Kaggle

## Title

OlpAI thử - Dự đoán vỡ nợ (dữ liệu bảng)

## Competition URL

pka-olpai-test-homecredit

## Subtitle

Dự đoán khách hàng vỡ nợ từ hồ sơ vay và lịch sử tín dụng ở nhiều bảng; test là các tháng sau train.

## Overview

Dự đoán khả năng vỡ nợ từ hồ sơ vay, lịch sử tín dụng ở tổ chức khác và các khoản vay trước (3 bảng nối bằng id). Train là 01/2019-06/2020, test là 07-12/2020. Chấm bằng AUROC.

## Description

Dạng bài **dữ liệu bảng**: dự đoán khả năng một khách hàng không trả được khoản vay, từ hồ sơ vay cùng lịch sử tín dụng ở các tổ chức khác và các khoản vay trước. Điểm nhấn: dữ liệu nằm ở **nhiều bảng nối bằng mã khách hàng** (phải tự gộp, tự tạo đặc trưng) và test là **các tháng sau** train.

Mỗi dòng của `application_*.csv` là một hồ sơ vay (thu nhập, khoản vay, gia đình, nghề nghiệp, nhà ở, điểm tín dụng bên ngoài...). `bureau.csv` ghi các khoản vay của khách hàng ở tổ chức tín dụng khác, `previous_application.csv` ghi các lần nộp hồ sơ vay trước đó tại Home Credit; cả hai nối với hồ sơ qua cột `id`. Khoảng 8% hồ sơ vỡ nợ.

**Nhiệm vụ**: với mỗi hồ sơ test, cho một điểm số, càng cao càng có khả năng vỡ nợ.

## Evaluation

**AUROC** (diện tích dưới đường ROC), thang 0-1: xác suất một hồ sơ vỡ nợ ngẫu nhiên có điểm cao hơn một hồ sơ không vỡ nợ ngẫu nhiên. Chỉ thứ tự điểm số có ý nghĩa, không cần hiệu chỉnh xác suất. Điểm bằng nhau tính một nửa. Nộp hằng số được 0,5.

## Submission File

`submission.csv` có tiêu đề `id,score`, mỗi hồ sơ test một dòng, `score` là số thực:

```
id,score
kh000123,0.0731
kh000456,0.2410
```

Public là các hồ sơ trong `application_test.csv` (07-09/2020), private là các hồ sơ trong `private_test.zip` (10-12/2020). Nên nộp đủ mọi dòng của cả hai; trước khi có mật khẩu, giữ nguyên các dòng private như `sample_submission.csv`.

## Timeline

- Bắt đầu: do người tổ chức thông báo.
- Giờ 0-5: làm bài, tối đa 20 lần nộp, chỉ có public test.
- Giờ thứ 6: công bố mật khẩu `private_test.zip`; dự đoán private, viết `main.py`, chọn bài nộp cuối.
- Kết thúc: 6 tiếng sau giờ bắt đầu; bảng private được mở ngay sau đó.

## Rules

- Không dùng dữ liệu ngoài bộ được cấp (kể cả dữ liệu cuộc thi Home Credit gốc trên Kaggle). Không gán nhãn tay cho test.
- Thư viện có sẵn: pandas, scikit-learn, XGBoost, LightGBM, CatBoost (nếu môi trường có).
- 5 tiếng đầu: tối đa 20 lần nộp public. Tiếng cuối: mật khẩu `private_test.zip` được công bố, tối đa 5 lần nộp private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Không mở `private_test.zip` trước khi mật khẩu được công bố.
- Tối đa 20 lần nộp; chọn bài nộp cuối để chấm private.

## Citation

Home Credit Group - Home Credit Default Risk (Kaggle, 2018). Bản Hugging Face mohameddhameem/home-credit-default-risk.

## Dataset Description

```
application_train.csv      153.614 hồ sơ (01/2019-06/2020): id, application_date, label (1 = vỡ nợ), 120 cột thông tin
application_test.csv       25.837 hồ sơ public (07-09/2020), như trên nhưng không có label
private_test.zip           (có mật khẩu) application_private_test.csv: 25.556 hồ sơ private (10-12/2020)
bureau.csv                 976.797 dòng: các khoản vay ở tổ chức khác, nối qua id (một khách nhiều dòng)
previous_application.csv   943.778 dòng: các hồ sơ vay trước tại Home Credit, nối qua id (một khách nhiều dòng)
sample_submission.csv      id,score cho mọi hồ sơ test (0,5)
baseline.ipynb, baseline.py
LICENSE-DATASET.txt
```

- Tên cột giữ như dữ liệu gốc Home Credit (ví dụ `AMT_CREDIT`, `DAYS_BIRTH`, `EXT_SOURCE_1..3`); các cột `DAYS_*` là số ngày tính lùi từ ngày nộp hồ sơ (số âm). Giá trị `365243` trong các cột `DAYS_*` của `previous_application` nghĩa là không có.
- `bureau.csv` và `previous_application.csv` chứa lịch sử của mọi khách hàng (train, public và private test).
- Mã khách hàng đã được đổi, không tra ngược được về dữ liệu gốc.

Giải nén private test (khi có mật khẩu):

```python
import zipfile
zipfile.ZipFile("private_test.zip").extractall(".", pwd=b"<mật khẩu>")   # hoặc: unzip -P <mật khẩu> private_test.zip
```

## Data license

Other (specified in description): dữ liệu cuộc thi Home Credit Default Risk, chỉ dùng học tập nội bộ, không phát tán lại.

## Files để tải lên

- Data: `HomeCredit_Tabular_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py` (hàm `auroc`), cao hơn là tốt hơn
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Mật khẩu private test: `host/mat_khau_private_test.txt`, công bố ở tiếng cuối
- Kiểm tra bằng sandbox submission: nộp `host/nop_dung_het.csv` (nếu có) phải ra điểm tối đa; nộp `submission.csv` của baseline phải gần `host/min_score.txt` (public 0.7612, private 0.7471)
