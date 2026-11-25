# nlp1_cam_xuc - nội dung từng ô trên Kaggle

## Title

OlpAI thử - NLP1 Cảm xúc phản hồi sinh viên

## Competition URL

olpai-thu5-cam-xuc

## Subtitle

Phân loại cảm xúc câu phản hồi của sinh viên: tích cực, tiêu cực, trung tính. Không mô hình huấn luyện sẵn.

## Overview

Phân loại cảm xúc câu phản hồi của sinh viên tiếng Việt thành 3 lớp, chấm bằng macro-F1. Không dùng mô hình ngôn ngữ huấn luyện sẵn.

## Description

Cho câu phản hồi của sinh viên về môn học, giảng viên (tiếng Việt, đã tách từ sơ bộ, chữ thường), phân loại cảm xúc: `positive`, `negative`, `neutral`.

Lớp `neutral` chỉ chiếm khoảng 4% nhưng có trọng số ngang hai lớp kia trong metric.

## Evaluation

**Macro-F1** trên 3 lớp.

```
id,label
te123456,positive
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

UIT-VSFC (Nguyễn Văn Kiệt và cộng sự, KSE 2018), CC BY-NC-SA 4.0.

## Dataset Description

- `train.csv` (`id, text, label`): 13.009 câu.
- `test.csv` (`id, text`): 3.166 câu.
- `sample_submission.csv`, `baseline.py` (chạy `python baseline.py /kaggle/input/olpai-thu5-cam-xuc`) (TF-IDF + logistic).

## Data license

CC BY-NC-SA 4.0

## Files để tải lên

- Data: `nlp1_cam_xuc_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py`
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Kiểm tra bằng sandbox submission: nộp file từ `baseline.py`, điểm phải gần `host/min_score.txt`
