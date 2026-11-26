# Dạng 9 - Phân loại văn bản (cảm xúc, chủ đề, ý định)

## 1. Nhận dạng đề

- Đầu vào: câu / đoạn văn (thường tiếng Việt), đầu ra: một nhãn (hoặc đa nhãn). Ví dụ: VSFC_Sentiment (phản hồi sinh viên, lớp trung tính rất hiếm), phân loại tin tức, ý định người dùng.
- Luật vòng khu vực: **không pretrained NLP** (không PhoBERT, không word embedding tải sẵn). `transformers` chỉ dùng kiến trúc khởi tạo ngẫu nhiên.

## 2. Metric

Macro-F1 thường gặp nhất. Chỉnh hệ số lớp trên OOF cho lớp hiếm.

## 3. Lộ trình

| Bước | Việc | Thời gian | Ghi chú |
|---|---|---|---|
| 0 | Baseline BTC, nộp | 10 phút | |
| 1 | NFC + chữ thường; **TF-IDF từ (1-2 gram) nối TF-IDF ký tự `char_wb` (2-5 gram)** + LogisticRegression / LinearSVC, CV 5-fold | 15 phút, CPU vài giây | thường đã rất mạnh, khó vượt |
| 2 | Chỉnh `C`, `class_weight="balanced"`, hệ số lớp trên OOF | 15 phút | |
| 3 | Mạng nơ-ron từ đầu: Embedding + BiGRU / TextCNN / Transformer nhỏ (2-4 lớp, d = 256) | 30-60 phút | |
| 4 | Ensemble xác suất TF-IDF + mạng nơ-ron (thường +1-2 điểm) | | |

## 4. Khung code

```python
import unicodedata, re
def sach(s):
    s = unicodedata.normalize("NFC", str(s)).lower()
    return re.sub(r"\s+", " ", s).strip()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
tu = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
kt = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300000)
Xtr = hstack([tu.fit_transform(tr), kt.fit_transform(tr)]).tocsr(); Xte = hstack([tu.transform(te), kt.transform(te)]).tocsr()
clf = LogisticRegression(C=4, max_iter=3000, class_weight="balanced").fit(Xtr, y)
```

(Trong CV, fit vectorizer **trong từng fold**; với TF-IDF không giám sát, fit trên toàn bộ train + test văn bản thường chấp nhận được nếu luật cho dùng văn bản test không nhãn.)

Mạng nơ-ron từ đầu:

```python
from collections import Counter
dem = Counter(w for s in train_texts for w in s.split())
itos = ["<pad>", "<unk>"] + [w for w, c in dem.items() if c >= 2]; stoi = {w: i for i, w in enumerate(itos)}
ma = lambda s, L=128: [stoi.get(w, 1) for w in s.split()][:L] or [1]

class BiGRU(nn.Module):
    def __init__(self, V, C, d=200, h=192):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=0); self.gru = nn.GRU(d, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3); self.fc = nn.Linear(4 * h, C)
    def forward(self, x):                                   # x: (B, L), 0 = pad
        o, _ = self.gru(self.drop(self.emb(x)))
        m = (x != 0).unsqueeze(-1)
        mean = (o * m).sum(1) / m.sum(1).clamp(min=1); mx = o.masked_fill(~m, -1e4).max(1).values
        return self.fc(self.drop(torch.cat([mean, mx], 1)))

batch = nn.utils.rnn.pad_sequence([torch.tensor(ma(s)) for s in texts], batch_first=True)   # sắp theo độ dài để ít pad
```

## 5. Nút vặn

1. Chuẩn hoá: NFC, chữ thường, teencode phổ biến ("ko" thành "không", "dc" thành "được") nếu dữ liệu mạng xã hội; giữ emoji / dấu "!" (mang cảm xúc).
2. N-gram ký tự (bền với lỗi gõ, mất dấu).
3. Tiếng Việt: ghép âm tiết thành từ ghép (bigram từ) thay cho bộ tách từ; bigram trong TF-IDF đã làm việc này.
4. Hệ số lớp / ngưỡng trên OOF cho macro-F1.
5. Ensemble khác loại (TF-IDF + mạng nơ-ron).
6. Pseudo-label văn bản không nhãn được cấp.

## 6. Bẫy

- NFC / NFD lẫn lộn (bài 28 luyện tay: quên NFC tụt từ 0,91 xuống 0,80).
- Câu rỗng / NaN làm TF-IDF lỗi: `fillna("")`.
- Nhãn văn bản ("positive") lẫn số: giữ một bảng ánh xạ duy nhất.

## 7. Bài luyện

- Code tay: [d28_nlp](../../luyen_tap/CV/d28_nlp/) phần A (TF-IDF + logistic) và A2 (TextCNN từ đầu).
- Thi thử: [VSFC_Sentiment](../../selfhost_kaggle/VSFC_Sentiment/) (baseline 0,664).
- Danh sách 16 bài NLP: [luyen_tap/NLP](../../luyen_tap/NLP/).

## 8. Đọc thêm

- Kim, *Convolutional Neural Networks for Sentence Classification* ([arXiv:1408.5882](https://arxiv.org/abs/1408.5882)).
- Joulin và cộng sự, *Bag of Tricks for Efficient Text Classification* (fastText, [arXiv:1607.01759](https://arxiv.org/abs/1607.01759)).
- Wang và Manning, *Baselines and Bigrams: Simple, Good Sentiment and Topic Classification* (NB-SVM, ACL 2012).
