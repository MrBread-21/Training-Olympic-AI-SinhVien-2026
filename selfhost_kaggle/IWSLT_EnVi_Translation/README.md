# NLP4 - Dịch máy Anh sang Việt

## Overview

Dịch câu nói trong các bài TED từ tiếng Anh sang tiếng Việt. Đề vòng sơ loại OlpAI 2025 là dịch Hoa sang Việt theo đúng thể thức này.

**Luật (giống vòng khu vực OlpAI'26)**:

- KHÔNG có pretrained NLP (BERT, PhoBERT, XLM-R, sentence-transformers, word2vec/fastText có sẵn...): mọi mô hình ngôn ngữ train từ đầu trên dữ liệu đề. `transformers` chỉ được dùng kiến trúc, không dùng weights.
- Không dùng dữ liệu ngoài.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Được dùng seq2seq/Transformer tự train, SMT, luật và từ điển tự xây từ dữ liệu train.

## Evaluation

**BLEU-4 mức corpus** (thang 0-100), cách tính như SacreBLEU: n-gram 1 đến 4 cắt theo tham chiếu, phạt độ ngắn. Trước khi chấm, câu được chuẩn hoá NFC, đổi chữ thường, tách từ và dấu câu.

```
id,vi
te123456,Khoa học đằng sau một tiêu đề về khí hậu
```

## Data

- `train.csv` (`id, en, vi`): 133.166 cặp câu.
- `test.csv` (`id, en`): 1.268 câu.
- `sample_submission.csv` (chép nguyên câu tiếng Anh), `baseline.py` (truy hồi câu train giống nhất rồi trả bản dịch của nó, không dịch).

Gợi ý thời gian: Transformer nhỏ (3 lớp, d=256) trên 133 nghìn cặp chạy vài epoch trong 1-2 giờ GPU.

Nguồn: IWSLT 2015 En-Vi (bản `thainq107/iwslt2015-en-vi`).
