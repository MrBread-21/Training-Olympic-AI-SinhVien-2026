# nlp4_dich - nội dung từng ô trên Kaggle

## Title

OlpAI thử - NLP4 Dịch Anh sang Việt

## Competition URL

olpai-thu5-dich

## Subtitle

Dịch câu nói TED từ tiếng Anh sang tiếng Việt, chấm BLEU. Không mô hình huấn luyện sẵn.

## Overview

Dịch máy Anh sang Việt trên dữ liệu TED (IWSLT 2015), chấm BLEU-4 như SacreBLEU. Không dùng mô hình dịch hay mô hình ngôn ngữ huấn luyện sẵn.

## Description

Dịch câu nói trong các bài TED từ tiếng Anh sang tiếng Việt. Đề vòng sơ loại OlpAI 2025 là dịch Hoa sang Việt theo đúng thể thức này.

## Evaluation

**BLEU-4 mức corpus** (thang 0-100), cách tính như SacreBLEU: n-gram 1 đến 4 cắt theo tham chiếu, phạt độ ngắn. Trước khi chấm, câu được chuẩn hoá NFC, đổi chữ thường, tách từ và dấu câu.

```
id,vi
te123456,Khoa học đằng sau một tiêu đề về khí hậu
```

## Timeline

- Bắt đầu: thứ 5, 26/11/2026 (giờ bắt đầu do người tổ chức thông báo).
- Kết thúc: 6 tiếng sau giờ bắt đầu. 5 tiếng đầu làm bài và nộp; tiếng cuối chốt mô hình, viết `main.py`, chọn 2 bài nộp cuối.
- Bảng private được mở ngay sau khi kết thúc.

## Rules

- KHÔNG có pretrained NLP (BERT, PhoBERT, XLM-R, sentence-transformers, word2vec/fastText có sẵn...): mọi mô hình ngôn ngữ train từ đầu trên dữ liệu đề. `transformers` chỉ được dùng kiến trúc, không dùng weights.
- Không dùng dữ liệu ngoài.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Được dùng seq2seq/Transformer tự train, SMT, luật và từ điển tự xây từ dữ liệu train.
- Không gán nhãn tay cho test.
- Tối đa 20 lần nộp; chọn 2 bài nộp cuối để chấm private.

## Citation

IWSLT 2015 En-Vi (bản `thainq107/iwslt2015-en-vi`).

## Dataset Description

- `train.csv` (`id, en, vi`): 133.166 cặp câu.
- `test.csv` (`id, en`): 1.268 câu.
- `sample_submission.csv` (chép nguyên câu tiếng Anh), `baseline.py` (chạy `python baseline.py /kaggle/input/olpai-thu5-dich`) (truy hồi câu train giống nhất rồi trả bản dịch của nó, không dịch).

Gợi ý thời gian: Transformer nhỏ (3 lớp, d=256) trên 133 nghìn cặp chạy vài epoch trong 1-2 giờ GPU.

## Data license

Other: IWSLT 2015 (TED Talks, CC BY-NC-ND), dùng cho nghiên cứu

## Files để tải lên

- Data: `nlp4_dich_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py`
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Kiểm tra bằng sandbox submission: nộp file từ `baseline.py`, điểm phải gần `host/min_score.txt`
