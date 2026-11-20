# Bài 28 (28/11) - NLP tiếng Việt: chuẩn hoá, TF-IDF, TextCNN, BM25

**Thời gian mục tiêu: ≤ 50 phút** (A 10, A2 25, B 15).

## Đọc trước khi làm

Đọc trong khoảng 30-45 phút trước khi bấm giờ; ghi ra giấy 3-5 ý chính (công thức, tham số, bẫy). Lúc làm bài thì đóng hết.

- **Unicode**: Python [unicodedata](https://docs.python.org/3/library/unicodedata.html) và [UAX #15 Unicode Normalization Forms](https://unicode.org/reports/tr15/), mục 1 (NFC so với NFD: cùng chữ "ệ" có thể là 1 hoặc 3 code point).
- **scikit-learn**: [Text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) (TfidfVectorizer, `analyzer="char_wb"`, `ngram_range`, `sublinear_tf`).
- **PyTorch**: [pad_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html), [nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) (`padding_idx`), [nn.Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) (đầu vào (B, C, L) nên cần `transpose(1, 2)`).
- **Bài báo**: Kim, *Convolutional Neural Networks for Sentence Classification* [arXiv:1408.5882](https://arxiv.org/abs/1408.5882), mục 2 (TextCNN, max-pool theo thời gian).
- **BM25**: Wikipedia [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) (công thức, ý nghĩa k1 và b); Manning và cộng sự, [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/), chương 6 (tf-idf) và mục 11.4.3 (BM25).
- **Trong máy**: `../../ON_TAP.md` phần 1.3; `../../../challenges/VietnameseExtract/solution.ipynb` (tokenize giữ vị trí ký tự, IDF; bản đó chưa chuẩn hoá NFC, bài này phải có).

Dữ liệu có bẫy giống văn bản thật: **train gần như toàn NFC nhưng khoảng 60% văn bản test và câu hỏi ở dạng NFD**; 20% câu mất dấu; lỗi gõ; từ của chủ đề khác chen vào.

## Phần A - Phân loại chủ đề bằng TF-IDF + logistic (`bai_a.py`)

- **Đầu vào**: `data/a_train.csv` (`id,text,label`, 3000 câu, 5 chủ đề), `data/a_test.csv` (`id,text`, 1000 câu).
- **Đầu ra**: `submission_a.csv` (`id,label`). **Metric**: macro-F1, **ngưỡng 0.87** (quên chuẩn hoá NFC chỉ được khoảng 0.80).
- **Khung**: `unicodedata.normalize("NFC", ...)` + chữ thường; TF-IDF từ (1-2 gram) nối với TF-IDF ký tự `char_wb` (2-5 gram); LogisticRegression; CV 5-fold, ma trận nhầm lẫn; fit lại toàn bộ.

## Phần A2 - Mạng nơ-ron tự train (`bai_a2.py`)

- **Đầu ra**: `submission_a2.csv`. **Ngưỡng 0.85**.
- **Khung**: tokenize (`re.findall(r"\w+")` sau NFC), **vocab** (`<pad>`, `<unk>`, tần suất ≥ 2), mã hoá; **bucketing** (sắp theo độ dài, batch câu gần dài nhau); TextCNN (Embedding, Conv1d k = 2/3/4, max-pool theo thời gian) hoặc BiLSTM; val ngoài vòng batch, lưu checkpoint tốt nhất.

## Phần B - Truy xuất bằng BM25 (`bai_b.py`)

- **Đầu vào**: `data/b_corpus.csv` (`id,text`, 400 đoạn giới thiệu công ty), `data/b_questions.csv` (`id,question`, 300 câu hỏi, không nêu tên công ty).
- **Đầu ra**: `submission_b.csv` (`id,passage_ids`), ít nhất 5 id đoạn văn, cách nhau bởi dấu cách. **Metric**: Recall@1, **ngưỡng 0.85**.
- **Khung**: chuẩn hoá NFC, chữ thường, **bỏ dấu** (câu hỏi có thể gõ không dấu); BM25 tự viết (IDF, k1 = 1.5, b = 0.75).

## Bài 2 - Viết lại nhanh bài 27 (≤ 35 phút)

Trong `d27_patchcore_unet`: `viet_lai_a.py`, `viet_lai_b.py`, chấm `python check_27.py viet_lai_a.py viet_lai_b.py`.

## Cách chạy

```
python tao_du_lieu.py
python check_28.py bai_a.py bai_a2.py bai_b.py
```
