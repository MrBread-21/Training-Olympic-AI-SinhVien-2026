# NLP2 - Nhận dạng thực thể trong tin tức COVID-19

## Overview

Cho câu tin tức tiếng Việt đã tách theo âm tiết, gán nhãn BIO cho từng âm tiết với 10 loại thực thể: `PATIENT_ID`, `NAME`, `AGE`, `GENDER`, `JOB`, `LOCATION`, `ORGANIZATION`, `SYMPTOM_AND_DISEASE`, `TRANSPORTATION`, `DATE`.

Ví dụ (từ train):

```
Bệnh nhân thứ 19 và 20 là bác gái và lái xe của Nhung .
O O O B-PATIENT_ID O B-PATIENT_ID O O B-GENDER O B-JOB I-JOB O B-NAME O
```

**Luật (giống vòng khu vực OlpAI'26)**:

- KHÔNG có pretrained NLP (BERT, PhoBERT, XLM-R, sentence-transformers, word2vec/fastText có sẵn...): mọi mô hình ngôn ngữ train từ đầu trên dữ liệu đề. `transformers` chỉ được dùng kiến trúc, không dùng weights.
- Không dùng dữ liệu ngoài.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Được dùng CRF, BiLSTM, Transformer tự train.

## Evaluation

**F1 mức thực thể** (micro): một thực thể chỉ đúng khi đúng cả biên và loại.

Cột `tags`: các nhãn cách nhau bởi dấu cách, **số nhãn bằng số âm tiết** trong `words` (tách bằng dấu cách). Thiếu nhãn thì phần thiếu bị coi là `O`.

```
id,tags
te123456,O O O B-PATIENT_ID O B-PATIENT_ID O O B-GENDER O B-JOB I-JOB O B-NAME O
```

## Data

- `train.csv` (`id, words, tags`): 7.027 câu.
- `test.csv` (`id, words`): 3.000 câu.
- `sample_submission.csv` (toàn O), `baseline.py` (mỗi âm tiết lấy nhãn hay gặp nhất).

Nguồn: PhoNER_COVID19 (Trương Thị Thu Hà và cộng sự, NAACL 2021), bản âm tiết.
