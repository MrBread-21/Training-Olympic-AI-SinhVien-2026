# Dạng 10 - Gán nhãn chuỗi (NER, POS, trích thông tin theo token)

## 1. Nhận dạng đề

- Đầu vào: câu đã tách token (âm tiết hoặc từ). Đầu ra: một nhãn cho **mỗi token**, thường dạng BIO (`B-PER`, `I-PER`, `O`).
- Ví dụ: PhoNER_COVID19 (tự host, 10 loại thực thể), trích khía cạnh / ý kiến trong DimABSA (xem [dạng 14](14_dimabsa.md)).
- File nộp: chuỗi nhãn cách nhau dấu cách, **cùng số token** với đầu vào.

## 2. Metric

F1 mức thực thể (seqeval): đúng khi trùng loại và cả hai biên. Xem [01_metric.md](../01_metric.md).

## 3. Lộ trình

| Bước | Việc | Ghi chú |
|---|---|---|
| 0 | Baseline BTC (thường "từ điển thực thể từ train"), nộp | |
| 1 | Từ điển + quy tắc: tra cụm dài nhất có trong danh sách thực thể train | nhanh, precision cao |
| 2 | **Embedding từ + embedding ký tự (CNN) + BiLSTM**, softmax từng token | train từ đầu vài phút |
| 3 | Thêm **CRF** (hoặc giải mã Viterbi có ràng buộc BIO) | +1-3 điểm, sửa chuỗi nhãn không hợp lệ |
| 4 | Đặc trưng thủ công: chữ hoa, chữ số, có trong từ điển train; ensemble | |

## 4. Khung code: BiLSTM + ký tự

```python
class Tagger(nn.Module):
    def __init__(self, Vw, Vc, T, dw=200, dc=50, h=256):
        super().__init__()
        self.ew = nn.Embedding(Vw, dw, padding_idx=0); self.ec = nn.Embedding(Vc, dc, padding_idx=0)
        self.cnn = nn.Conv1d(dc, 100, 3, padding=1)
        self.lstm = nn.LSTM(dw + 100, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.drop = nn.Dropout(0.3); self.fc = nn.Linear(2 * h, T)
    def forward(self, w, c):                       # w: (B, L) id từ; c: (B, L, K) id ký tự của từng từ
        B, L, K = c.shape
        ce = self.ec(c.view(B * L, K)).transpose(1, 2)                  # (B*L, dc, K)
        ce = self.cnn(ce).max(2).values.view(B, L, 100)                 # đặc trưng ký tự mỗi từ
        o, _ = self.lstm(self.drop(torch.cat([self.ew(w), ce], -1)))
        return self.fc(self.drop(o))                                    # (B, L, T)
loss = F.cross_entropy(logits.reshape(-1, T), tags.reshape(-1), ignore_index=-100)   # tag của pad = -100
```

Giải mã Viterbi có ràng buộc BIO (không cần thư viện CRF): cấm chuyển `O` sang `I-X` và `B-Y` / `I-Y` sang `I-X` khi X khác Y.

```python
def viterbi_bio(logp, tags):                       # logp: (L, T) log-softmax của một câu; tags: list tên nhãn
    T = len(tags); cam = np.zeros((T, T))
    for i, a in enumerate(tags):
        for j, b in enumerate(tags):
            if b.startswith("I-") and not (a != "O" and a[2:] == b[2:]): cam[i, j] = -1e9
    bat_dau = np.array([-1e9 if t.startswith("I-") else 0 for t in tags])
    d = logp[0] + bat_dau; truoc = []
    for t in range(1, len(logp)):
        s = d[:, None] + cam + logp[t][None, :]; truoc.append(s.argmax(0)); d = s.max(0)
    duong = [int(d.argmax())]
    for p in reversed(truoc): duong.append(int(p[duong[-1]]))
    return [tags[i] for i in duong[::-1]]
```

## 5. Nút vặn

1. Embedding ký tự (từ hiếm, tên riêng, chữ hoa).
2. Viterbi / CRF.
3. Từ điển thực thể train làm đặc trưng (cờ "token này nằm trong một thực thể đã biết").
4. Dropout trên embedding, word dropout (thay ngẫu nhiên từ bằng `<unk>` 5-10%) để học xử lý từ lạ.
5. Ensemble 3-5 seed (bỏ phiếu theo token rồi Viterbi).

## 6. Bẫy

- Số nhãn nộp khác số token: luôn kiểm tra `len(pred) == len(words)` từng câu.
- Câu dài bị cắt khi đệm: dự đoán phần bị cắt là `O` sẽ mất thực thể; đặt độ dài tối đa đủ lớn hoặc chia đoạn.
- Âm tiết hay từ: PhoNER có bản theo âm tiết và theo từ, tokenizer phải khớp.

## 7. Bài luyện

- Thi thử: [PhoNER_COVID19](../../selfhost_kaggle/PhoNER_COVID19/) (baseline 0,428).
- Tự làm: viết `entity_f1` ([metric_tu_viet.py](../metric_tu_viet.py)) và kiểm tra với seqeval nếu có.

## 8. Đọc thêm

- Lample và cộng sự, *Neural Architectures for Named Entity Recognition* ([arXiv:1603.01360](https://arxiv.org/abs/1603.01360)).
- Ma và Hovy, *End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF* ([arXiv:1603.01354](https://arxiv.org/abs/1603.01354)).
- Truong và cộng sự, *COVID-19 Named Entity Recognition for Vietnamese* (PhoNER, [arXiv:2104.03879](https://arxiv.org/abs/2104.03879)).
