# Dự đoán vỡ nợ (Home Credit, dữ liệu bảng nhiều file)

Dạng bài **dữ liệu bảng**: dự đoán khả năng một khách hàng không trả được khoản vay, từ hồ sơ vay cùng lịch sử tín dụng ở các tổ chức khác và các khoản vay trước. Điểm nhấn: dữ liệu nằm ở **nhiều bảng nối bằng mã khách hàng** (phải tự gộp, tự tạo đặc trưng) và test là **các tháng sau** train.

## Overview

Mỗi dòng của `application_*.csv` là một hồ sơ vay (thu nhập, khoản vay, gia đình, nghề nghiệp, nhà ở, điểm tín dụng bên ngoài...). `bureau.csv` ghi các khoản vay của khách hàng ở tổ chức tín dụng khác, `previous_application.csv` ghi các lần nộp hồ sơ vay trước đó tại Home Credit; cả hai nối với hồ sơ qua cột `id`. Khoảng 8% hồ sơ vỡ nợ.

**Nhiệm vụ**: với mỗi hồ sơ test, cho một điểm số, càng cao càng có khả năng vỡ nợ.

**Luật (giống vòng khu vực OlpAI'26)**:

- Không dùng dữ liệu ngoài bộ được cấp (kể cả dữ liệu cuộc thi Home Credit gốc trên Kaggle). Không gán nhãn tay cho test.
- Thư viện có sẵn: pandas, scikit-learn, XGBoost, LightGBM, CatBoost (nếu môi trường có).
- 5 tiếng đầu: tối đa 20 lần nộp public. Tiếng cuối: mật khẩu `private_test.zip` được công bố, tối đa 5 lần nộp private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.

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

## Data

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

## Hướng làm gợi ý

- Validation theo thời gian (vài tháng cuối của train), không xáo ngẫu nhiên.
- Gộp bảng phụ theo `id`: số khoản vay, tổng / trung bình / lớn nhất của số tiền, tỉ lệ khoản đang nợ, số ngày quá hạn, tỉ lệ hồ sơ trước bị từ chối...
- Đặc trưng tỉ lệ: khoản vay / thu nhập, tiền trả hằng kỳ / thu nhập, tuổi, số năm đi làm.
- LightGBM / XGBoost / CatBoost với early stopping, rồi blend.

Nguồn: Home Credit Default Risk (Kaggle 2018), bản Hugging Face `mohameddhameem/home-credit-default-risk` có thêm cột ngày nộp hồ sơ. Dùng ba bảng: application, bureau, previous_application.
