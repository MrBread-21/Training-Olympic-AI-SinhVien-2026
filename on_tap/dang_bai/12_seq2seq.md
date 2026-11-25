# Dạng 12 - Sinh chuỗi: dịch máy, thêm dấu, chuẩn hoá văn bản

## 1. Nhận dạng đề

- Câu nguồn thành câu đích: dịch Trung - Việt (vòng khu vực 2025, SacreBLEU, cấm model dịch pretrained), dịch Anh - Việt (IWSLT_EnVi_Translation tự host), thêm dấu tiếng Việt (Vietnamese_Diacritics tự host), sửa chính tả, chuẩn hoá teencode.
- Đặc biệt: **thêm dấu** có đầu ra cùng độ dài với đầu vào (mỗi âm tiết không dấu thành một âm tiết có dấu), nên làm như **gán nhãn chuỗi**, không cần seq2seq.

## 2. Metric

BLEU-4 (SacreBLEU), chrF, độ chính xác từ (thêm dấu). Xem [01_metric.md](../01_metric.md).

## 3. Lộ trình (dịch máy từ đầu)

| Bước | Việc | Ghi chú |
|---|---|---|
| 0 | Baseline BTC, nộp | |
| 1 | Làm sạch: NFC, bỏ cặp quá dài (> 100 token) hoặc lệch độ dài (tỉ lệ > 3), bỏ trùng | |
| 2 | Tokenizer: tiếng Việt / Anh theo từ (đã tách bằng dấu cách) hoặc **BPE** bằng thư viện `tokenizers` (train trên dữ liệu bài, vocab 8-16k); tiếng Trung theo **ký tự** | |
| 3 | `nn.Transformer` nhỏ: 3-4 lớp encoder / decoder, d_model 256-512, 4-8 head, dropout 0,1-0,3, label smoothing 0,1, warmup + inverse-sqrt | BLEU có sau 30-45 phút train |
| 4 | Greedy làm mốc, rồi **beam 4-5 + length penalty**, giới hạn độ dài theo câu nguồn | +1-2 BLEU |
| 5 | **Trung bình trọng số 3-5 checkpoint cuối**, ensemble | +0,5-1 BLEU |

## 4. Khung code

```python
class Seq2Seq(nn.Module):
    def __init__(self, Vs, Vt, d=384, L=4, h=6, ff=1536, p=0.2, maxlen=256):
        super().__init__()
        self.es, self.et, self.pos = nn.Embedding(Vs, d, padding_idx=0), nn.Embedding(Vt, d, padding_idx=0), nn.Embedding(maxlen, d)
        self.tf = nn.Transformer(d, h, L, L, ff, p, batch_first=True, norm_first=True)
        self.out = nn.Linear(d, Vt); self.out.weight = self.et.weight          # chia sẻ embedding đầu ra
        self.d = d
    def them_vt(self, e, x): return e(x) * self.d ** 0.5 + self.pos(torch.arange(x.shape[1], device=x.device))
    def forward(self, src, tgt_in):
        m = nn.Transformer.generate_square_subsequent_mask(tgt_in.shape[1], device=src.device)
        h = self.tf(self.them_vt(self.es, src), self.them_vt(self.et, tgt_in), tgt_mask=m,
                    src_key_padding_mask=src == 0, tgt_key_padding_mask=tgt_in == 0, memory_key_padding_mask=src == 0)
        return self.out(h)

# train: tgt = [BOS] + câu + [EOS]; vào decoder tgt[:, :-1], nhãn tgt[:, 1:]
loss = F.cross_entropy(logits.reshape(-1, Vt), tgt[:, 1:].reshape(-1), ignore_index=0, label_smoothing=0.1)
opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)   # lr thật do LambdaLR quyết định
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 384 ** -0.5 * min((s + 1) ** -0.5, (s + 1) * 4000 ** -1.5))   # warmup 4000 bước rồi giảm theo 1/căn(bước); gọi sch.step() mỗi batch
```

Beam search (gọn, cho từng câu; chạy theo batch nhanh hơn nhưng dài hơn):

```python
@torch.inference_mode()
def beam(model, src, k=5, alpha=0.6, maxlen=None):
    maxlen = maxlen or int(src.shape[1] * 1.5) + 10
    mem = model.tf.encoder(model.them_vt(model.es, src), src_key_padding_mask=src == 0)
    chum = [([BOS], 0.0)]
    for _ in range(maxlen):
        moi = []
        for seq, sc in chum:
            if seq[-1] == EOS: moi.append((seq, sc)); continue
            t = torch.tensor([seq], device=src.device)
            h = model.tf.decoder(model.them_vt(model.et, t), mem, tgt_mask=nn.Transformer.generate_square_subsequent_mask(len(seq), device=src.device),
                                 memory_key_padding_mask=src == 0)
            lp = model.out(h[:, -1]).log_softmax(-1)[0]
            lp[0] = lp[BOS] = -float("inf")                       # không bao giờ sinh PAD (0) hay BOS
            v, i = lp.topk(k)
            moi += [(seq + [int(j)], sc + float(x)) for x, j in zip(v, i)]
        chum = sorted(moi, key=lambda z: z[1] / (((5 + len(z[0])) / 6) ** alpha), reverse=True)[:k]   # length penalty GNMT
        if all(s[-1] == EOS for s, _ in chum): break
    return chum[0][0]
```

Trung bình checkpoint:

```python
sds = [torch.load(f"ckpt_{e}.pt") for e in last_epochs]
avg = {k: sum(sd[k].float() for sd in sds) / len(sds) for k in sds[0]}
model.load_state_dict(avg)
```

Thêm dấu: mỗi âm tiết không dấu là một token đầu vào, nhãn là âm tiết có dấu (hoặc "kiểu dấu" của âm tiết). BiLSTM / Transformer encoder + softmax từng vị trí, **chỉ cho phép các âm tiết có dấu ứng với đúng âm tiết không dấu đó** (mặt nạ ứng viên lấy từ train). Mốc mạnh không cần mạng: chọn dạng có dấu phổ biến nhất của mỗi bigram / trigram âm tiết (n-gram + backoff).

## 5. Nút vặn

1. Làm sạch dữ liệu và lọc cặp lệch.
2. BPE / vocab vừa phải; chia sẻ embedding.
3. Label smoothing, dropout, warmup.
4. Beam + length penalty; giới hạn độ dài đầu ra.
5. Trung bình checkpoint; ensemble 2 mô hình.
6. Đánh giá bằng đúng cách tách từ của bộ chấm (SacreBLEU tự tách).

## 6. Bẫy

- Quên dịch chuyển (shift) giữa đầu vào decoder và nhãn: loss giảm rất nhanh nhưng giải mã ra rác.
- `PAD` trùng chỉ số với token thật; `padding_idx` và `ignore_index` phải cùng giá trị.
- Dự đoán lặp từ vô hạn: giới hạn độ dài, chặn lặp n-gram.
- SacreBLEU trên văn bản đã tách từ khác với văn bản gốc; nộp văn bản **tự nhiên** (ghép lại dấu câu) nếu đề yêu cầu.

## 7. Bài luyện

- Code tay: [d29_seq2seq](../../luyen_tap/CV/d29_seq2seq/) (`nn.Transformer` từ đầu, mặt nạ, greedy + beam, BLEU / chrF).
- Thi thử: [IWSLT_EnVi_Translation](../../selfhost_kaggle/IWSLT_EnVi_Translation/) (baseline BLEU 2,32), [Vietnamese_Diacritics](../../selfhost_kaggle/Vietnamese_Diacritics/) (baseline 0,754).

## 8. Đọc thêm

- Vaswani và cộng sự, *Attention Is All You Need* ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762)); Rush, [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).
- Sennrich và cộng sự, *Neural Machine Translation of Rare Words with Subword Units* (BPE, [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)).
- Post, *A Call for Clarity in Reporting BLEU Scores* (SacreBLEU, [arXiv:1804.08771](https://arxiv.org/abs/1804.08771)).
