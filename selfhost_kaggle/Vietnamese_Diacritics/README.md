# NLP3 - Thêm dấu tiếng Việt

## Overview

Cho câu tiếng Việt đã bị bỏ hết dấu (kể cả `đ` thành `d`), khôi phục câu có dấu. Câu lấy từ tiêu đề và tóm tắt báo điện tử.

Ví dụ: `Hai nhan vien gac chan tau bi phat 1,5 trieu dong` thành `Hai nhân viên gác chắn tàu bị phạt 1,5 triệu đồng`.

Cùng một âm tiết không dấu có nhiều cách thêm dấu (`ban` có thể là bán, bàn, bạn, bản...), phải dựa vào ngữ cảnh.

**Luật (giống vòng khu vực OlpAI'26)**:

- KHÔNG có pretrained NLP (BERT, PhoBERT, XLM-R, sentence-transformers, word2vec/fastText có sẵn...): mọi mô hình ngôn ngữ train từ đầu trên dữ liệu đề. `transformers` chỉ được dùng kiến trúc, không dùng weights.
- Không dùng dữ liệu ngoài.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.

## Evaluation

**Độ chính xác theo từ**: tách câu đáp án và câu nộp theo dấu cách, so từng vị trí. Từ thiếu tính là sai. Giữ nguyên số từ, hoa thường và dấu câu như đầu vào.

```
id,text
te123456,Hai nhân viên gác chắn tàu bị phạt 1,5 triệu đồng
```

## Data

- `train.csv` (`id, text_khong_dau, text`): 100.000 câu.
- `test.csv` (`id, text_khong_dau`): 5.000 câu.
- `sample_submission.csv` (giữ nguyên câu không dấu), `baseline.py` (mỗi từ lấy dạng có dấu hay gặp nhất).

Nguồn: `NamSyntax/vietnamese-news-classification` (Apache 2.0).
