# Dạng 13 - Truy xuất văn bản (tìm đoạn văn, tìm tài liệu)

## 1. Nhận dạng đề

- Kho văn bản (corpus) + câu hỏi; với mỗi câu hỏi trả về K đoạn văn xếp hạng. Ví dụ: ViQuAD_PassageRetrieval (tự host, MRR@10), bước đầu của hỏi đáp nhiều đoạn, tìm sản phẩm theo mô tả.
- Không pretrained: không có bi-encoder sẵn; BM25 là xương sống.

## 2. Metric

Recall@K, MRR@K, nDCG. Xem [01_metric.md](../01_metric.md).

## 3. Lộ trình

| Bước | Việc | Ghi chú |
|---|---|---|
| 0 | Baseline BTC, nộp | |
| 1 | **BM25 tự viết** trên âm tiết sau NFC + chữ thường (k1 = 1,2-1,5, b = 0,75) | thường rất mạnh |
| 2 | Thêm bigram âm tiết, bản bỏ dấu (câu hỏi gõ không dấu), ghép tiêu đề vào đoạn văn | |
| 3 | Kết hợp BM25 + TF-IDF ký tự (cosine) bằng tổng hạng / tổng điểm chuẩn hoá | |
| 4 | **Re-rank top 50** bằng mô hình học từ cặp train (LightGBM / logistic trên đặc trưng: điểm BM25 từng trường, số từ trùng, IDF trùng lớn nhất, độ phủ bigram, khoảng cách giữa các từ trùng) | |
| 5 | Bi-encoder train từ đầu (in-batch negative) nếu dữ liệu đủ lớn | |

## 4. Khung code: BM25 bằng ma trận thưa

```python
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
def bm25_fit(docs, k1=1.5, b=0.75):
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 1))
    tf = cv.fit_transform(docs).tocsr().astype(np.float32)               # (N, V)
    N, dl = tf.shape[0], np.asarray(tf.sum(1)).ravel()
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    tf.data = tf.data * (k1 + 1) / (tf.data + np.repeat(k1 * (1 - b + b * dl / dl.mean()), np.diff(tf.indptr)))
    return cv, (tf @ sp.diags(idf)).T.tocsr()                            # (V, N) trọng số đã nhân IDF
def bm25_score(cv, W, queries):
    q = cv.transform(queries); q.data[:] = 1                             # mỗi từ của câu hỏi tính một lần
    return (q @ W).toarray()                                             # (Nq, N)
top10 = np.argsort(-S, 1)[:, :10]
```

Bỏ dấu tiếng Việt (cho bản chỉ mục phụ):

```python
def bo_dau(s):
    s = unicodedata.normalize("NFD", s).replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", "".join(c for c in s if unicodedata.category(c) != "Mn"))
```

## 5. Nút vặn

1. Chuẩn hoá (NFC, chữ thường), giữ chữ số.
2. Bigram âm tiết (từ ghép tiếng Việt) như một trường thứ hai; bản bỏ dấu.
3. k1, b chọn trên train (có cặp câu hỏi, đoạn đúng).
4. Re-rank bằng mô hình học; đặc trưng "đoạn văn thuộc bài có tiêu đề trùng câu hỏi".
5. Kết hợp nhiều bộ chấm bằng hạng (reciprocal rank fusion: `tổng 1 / (60 + hạng)`).

## 6. Bẫy

- Câu hỏi không nêu đúng tên thực thể (đại từ, mô tả gián tiếp): BM25 kém, cần đặc trưng ngữ nghĩa từ dữ liệu train.
- Dùng `CountVectorizer` mặc định bỏ token 1 ký tự: đặt `token_pattern` để giữ.
- Ma trận điểm dày `Nq x N` lớn: chia khối câu hỏi.

## 7. Bài luyện

- Code tay: [d28_nlp](../../luyen_tap/CV/d28_nlp/) phần B (BM25 tự viết, bỏ dấu).
- Thi thử: [ViQuAD_PassageRetrieval](../../selfhost_kaggle/ViQuAD_PassageRetrieval/) (baseline 0,544).

## 8. Đọc thêm

- Manning, Raghavan, Schütze, [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/), chương 6 và mục 11.4.3 (BM25).
- Cormack và cộng sự, *Reciprocal Rank Fusion* (SIGIR 2009).
- Karpukhin và cộng sự, *Dense Passage Retrieval* ([arXiv:2004.04906](https://arxiv.org/abs/2004.04906)): in-batch negative.
