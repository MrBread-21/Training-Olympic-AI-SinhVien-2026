# nlp3_them_dau - nội dung từng ô trên Kaggle

## Title

OlpAI thử - NLP3 Thêm dấu tiếng Việt

## Competition URL

olpai-thu5-them-dau

## Subtitle

Khôi phục dấu cho câu tiếng Việt đã bị bỏ dấu. Không mô hình huấn luyện sẵn.

## Overview

Khôi phục dấu cho câu tiếng Việt đã bị bỏ hết dấu, chấm độ chính xác theo từ. Không dùng mô hình ngôn ngữ huấn luyện sẵn.

## Description

Cho câu tiếng Việt đã bị bỏ hết dấu (kể cả `đ` thành `d`), khôi phục câu có dấu. Câu lấy từ tiêu đề và tóm tắt báo điện tử.

Ví dụ: `Hai nhan vien gac chan tau bi phat 1,5 trieu dong` thành `Hai nhân viên gác chắn tàu bị phạt 1,5 triệu đồng`.

Cùng một âm tiết không dấu có nhiều cách thêm dấu (`ban` có thể là bán, bàn, bạn, bản...), phải dựa vào ngữ cảnh.

## Evaluation

**Độ chính xác theo từ**: tách câu đáp án và câu nộp theo dấu cách, so từng vị trí. Từ thiếu tính là sai. Giữ nguyên số từ, hoa thường và dấu câu như đầu vào.

```
id,text
te123456,Hai nhân viên gác chắn tàu bị phạt 1,5 triệu đồng
```

## Timeline

- Bắt đầu: thứ 5, 26/11/2026 (giờ bắt đầu do người tổ chức thông báo).
- Kết thúc: 6 tiếng sau giờ bắt đầu. 5 tiếng đầu làm bài và nộp; tiếng cuối chốt mô hình, viết `main.py`, chọn 2 bài nộp cuối.
- Bảng private được mở ngay sau khi kết thúc.

## Rules

- KHÔNG có pretrained NLP (BERT, PhoBERT, XLM-R, sentence-transformers, word2vec/fastText có sẵn...): mọi mô hình ngôn ngữ train từ đầu trên dữ liệu đề. `transformers` chỉ được dùng kiến trúc, không dùng weights.
- Không dùng dữ liệu ngoài.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Không gán nhãn tay cho test.
- Tối đa 20 lần nộp; chọn 2 bài nộp cuối để chấm private.

## Citation

`NamSyntax/vietnamese-news-classification` (Apache 2.0).

## Dataset Description

- `train.csv` (`id, text_khong_dau, text`): 100.000 câu.
- `test.csv` (`id, text_khong_dau`): 5.000 câu.
- `sample_submission.csv` (giữ nguyên câu không dấu), `baseline.py` (chạy `python baseline.py /kaggle/input/olpai-thu5-them-dau`) (mỗi từ lấy dạng có dấu hay gặp nhất).

## Data license

Apache 2.0

## Files để tải lên

- Data: `nlp3_them_dau_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py`
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Kiểm tra bằng sandbox submission: nộp file từ `baseline.py`, điểm phải gần `host/min_score.txt`
