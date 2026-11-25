# nlp2_ner - nội dung từng ô trên Kaggle

## Title

OlpAI thử - NLP2 Nhận dạng thực thể

## Competition URL

olpai-thu5-ner

## Subtitle

Gán nhãn BIO cho 10 loại thực thể trong tin tức COVID-19 tiếng Việt. Không mô hình huấn luyện sẵn.

## Overview

Nhận dạng 10 loại thực thể (bệnh nhân, địa điểm, ngày, triệu chứng...) trong tin tức COVID-19 tiếng Việt, chấm F1 mức thực thể. Không dùng mô hình ngôn ngữ huấn luyện sẵn.

## Description

Cho câu tin tức tiếng Việt đã tách theo âm tiết, gán nhãn BIO cho từng âm tiết với 10 loại thực thể: `PATIENT_ID`, `NAME`, `AGE`, `GENDER`, `JOB`, `LOCATION`, `ORGANIZATION`, `SYMPTOM_AND_DISEASE`, `TRANSPORTATION`, `DATE`.

Ví dụ (từ train):

```
Bệnh nhân thứ 19 và 20 là bác gái và lái xe của Nhung .
O O O B-PATIENT_ID O B-PATIENT_ID O O B-GENDER O B-JOB I-JOB O B-NAME O
```

## Evaluation

**F1 mức thực thể** (micro): một thực thể chỉ đúng khi đúng cả biên và loại.

Cột `tags`: các nhãn cách nhau bởi dấu cách, **số nhãn bằng số âm tiết** trong `words` (tách bằng dấu cách). Thiếu nhãn thì phần thiếu bị coi là `O`.

```
id,tags
te123456,O O O B-PATIENT_ID O B-PATIENT_ID O O B-GENDER O B-JOB I-JOB O B-NAME O
```

## Timeline

- Bắt đầu: thứ 5, 26/11/2026 (giờ bắt đầu do người tổ chức thông báo).
- Kết thúc: 6 tiếng sau giờ bắt đầu. 5 tiếng đầu làm bài và nộp; tiếng cuối chốt mô hình, viết `main.py`, chọn 2 bài nộp cuối.
- Bảng private được mở ngay sau khi kết thúc.

## Rules

- KHÔNG có pretrained NLP (BERT, PhoBERT, XLM-R, sentence-transformers, word2vec/fastText có sẵn...): mọi mô hình ngôn ngữ train từ đầu trên dữ liệu đề. `transformers` chỉ được dùng kiến trúc, không dùng weights.
- Không dùng dữ liệu ngoài.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Được dùng CRF, BiLSTM, Transformer tự train.
- Không gán nhãn tay cho test.
- Tối đa 20 lần nộp; chọn 2 bài nộp cuối để chấm private.

## Citation

PhoNER_COVID19 (Trương Thị Thu Hà và cộng sự, NAACL 2021), bản âm tiết.

## Dataset Description

- `train.csv` (`id, words, tags`): 7.027 câu.
- `test.csv` (`id, words`): 3.000 câu.
- `sample_submission.csv` (toàn O), `baseline.py` (chạy `python baseline.py /kaggle/input/olpai-thu5-ner`) (mỗi âm tiết lấy nhãn hay gặp nhất).

## Data license

Other: PhoNER_COVID19, dùng cho nghiên cứu

## Files để tải lên

- Data: `nlp2_ner_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py`
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Kiểm tra bằng sandbox submission: nộp file từ `baseline.py`, điểm phải gần `host/min_score.txt`
