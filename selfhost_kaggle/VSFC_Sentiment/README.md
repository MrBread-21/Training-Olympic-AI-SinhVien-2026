# NLP1 - Phân loại cảm xúc phản hồi của sinh viên

## Overview

Cho câu phản hồi của sinh viên về môn học, giảng viên (tiếng Việt, đã tách từ sơ bộ, chữ thường), phân loại cảm xúc: `positive`, `negative`, `neutral`.

Lớp `neutral` chỉ chiếm khoảng 4% nhưng có trọng số ngang hai lớp kia trong metric.

**Luật (giống vòng khu vực OlpAI'26)**:

- KHÔNG có pretrained NLP (BERT, PhoBERT, XLM-R, sentence-transformers, word2vec/fastText có sẵn...): mọi mô hình ngôn ngữ train từ đầu trên dữ liệu đề. `transformers` chỉ được dùng kiến trúc, không dùng weights.
- Không dùng dữ liệu ngoài.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.

## Evaluation

**Macro-F1** trên 3 lớp.

```
id,label
te123456,positive
```

## Data

- `train.csv` (`id, text, label`): 13.009 câu.
- `test.csv` (`id, text`): 3.166 câu.
- `sample_submission.csv`, `baseline.py` (TF-IDF + logistic).

Nguồn: UIT-VSFC (Nguyễn Văn Kiệt và cộng sự, KSE 2018), CC BY-NC-SA 4.0.
