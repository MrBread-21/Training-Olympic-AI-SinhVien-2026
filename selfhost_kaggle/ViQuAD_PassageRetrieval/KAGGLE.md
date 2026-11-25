# nlp5_truy_xuat - nội dung từng ô trên Kaggle

## Title

OlpAI thử - NLP5 Truy xuất đoạn văn

## Competition URL

olpai-thu5-truy-xuat

## Subtitle

Tìm đoạn Wikipedia tiếng Việt chứa câu trả lời cho câu hỏi, chấm MRR@10. Không mô hình huấn luyện sẵn.

## Overview

Tìm trong 5.317 đoạn Wikipedia tiếng Việt đoạn chứa câu trả lời cho câu hỏi, chấm MRR@10. Không dùng mô hình ngôn ngữ huấn luyện sẵn.

## Description

Cho một kho 5.317 đoạn văn Wikipedia tiếng Việt và câu hỏi, tìm đoạn văn chứa câu trả lời. Đây là bước truy xuất trong hệ thống hỏi đáp.

Đoạn văn của câu hỏi test có thể không xuất hiện trong bất kỳ câu hỏi train nào, nên không học thuộc được ánh xạ câu hỏi sang đoạn.

## Evaluation

**MRR@10**: với mỗi câu hỏi, nếu đoạn đúng ở vị trí k (k ≤ 10) thì được 1/k, không có thì 0; lấy trung bình.

Cột `passage_ids`: tối đa 10 id đoạn văn cách nhau bởi dấu cách, xếp từ khớp nhất.

```
id,passage_ids
q123456,p004211 p118273 p903311
```

## Timeline

- Bắt đầu: thứ 5, 26/11/2026 (giờ bắt đầu do người tổ chức thông báo).
- Kết thúc: 6 tiếng sau giờ bắt đầu. 5 tiếng đầu làm bài và nộp; tiếng cuối chốt mô hình, viết `main.py`, chọn 2 bài nộp cuối.
- Bảng private được mở ngay sau khi kết thúc.

## Rules

- KHÔNG có pretrained NLP (BERT, PhoBERT, XLM-R, sentence-transformers, word2vec/fastText có sẵn...): mọi mô hình ngôn ngữ train từ đầu trên dữ liệu đề. `transformers` chỉ được dùng kiến trúc, không dùng weights.
- Không dùng dữ liệu ngoài.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Được dùng BM25, TF-IDF, mô hình tự train trên `train.csv` (ví dụ học xếp hạng lại).
- Không gán nhãn tay cho test.
- Tối đa 20 lần nộp; chọn 2 bài nộp cuối để chấm private.

## Citation

UIT-ViQuAD 2.0 (Nguyễn Văn Kiệt và cộng sự), bản `taidng/UIT-ViQuAD2.0`.

## Dataset Description

- `corpus.csv` (`passage_id, title, text`): 5.317 đoạn văn.
- `train.csv` (`id, question, passage_id`): 28.454 câu hỏi có đoạn đúng.
- `test.csv` (`id, question`): 4.000 câu hỏi.
- `sample_submission.csv` (ngẫu nhiên), `baseline.py` (chạy `python baseline.py /kaggle/input/olpai-thu5-truy-xuat`) (TF-IDF cosine).

## Data license

Other: UIT-ViQuAD 2.0 (nội dung Wikipedia CC BY-SA), dùng cho nghiên cứu

## Files để tải lên

- Data: `nlp5_truy_xuat_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py`
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Kiểm tra bằng sandbox submission: nộp file từ `baseline.py`, điểm phải gần `host/min_score.txt`
