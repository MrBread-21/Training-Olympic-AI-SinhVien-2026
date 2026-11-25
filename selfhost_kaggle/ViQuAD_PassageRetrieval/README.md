# NLP5 - Truy xuất đoạn văn trả lời câu hỏi

## Overview

Cho một kho 5.317 đoạn văn Wikipedia tiếng Việt và câu hỏi, tìm đoạn văn chứa câu trả lời. Đây là bước truy xuất trong hệ thống hỏi đáp.

Đoạn văn của câu hỏi test có thể không xuất hiện trong bất kỳ câu hỏi train nào, nên không học thuộc được ánh xạ câu hỏi sang đoạn.

**Luật (giống vòng khu vực OlpAI'26)**:

- KHÔNG có pretrained NLP (BERT, PhoBERT, XLM-R, sentence-transformers, word2vec/fastText có sẵn...): mọi mô hình ngôn ngữ train từ đầu trên dữ liệu đề. `transformers` chỉ được dùng kiến trúc, không dùng weights.
- Không dùng dữ liệu ngoài.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Được dùng BM25, TF-IDF, mô hình tự train trên `train.csv` (ví dụ học xếp hạng lại).

## Evaluation

**MRR@10**: với mỗi câu hỏi, nếu đoạn đúng ở vị trí k (k ≤ 10) thì được 1/k, không có thì 0; lấy trung bình.

Cột `passage_ids`: tối đa 10 id đoạn văn cách nhau bởi dấu cách, xếp từ khớp nhất.

```
id,passage_ids
q123456,p004211 p118273 p903311
```

## Data

- `corpus.csv` (`passage_id, title, text`): 5.317 đoạn văn.
- `train.csv` (`id, question, passage_id`): 28.454 câu hỏi có đoạn đúng.
- `test.csv` (`id, question`): 4.000 câu hỏi.
- `sample_submission.csv` (ngẫu nhiên), `baseline.py` (TF-IDF cosine).

Nguồn: UIT-ViQuAD 2.0 (Nguyễn Văn Kiệt và cộng sự), bản `taidng/UIT-ViQuAD2.0`.
