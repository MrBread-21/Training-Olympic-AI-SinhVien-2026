# Dạng 7 - Sinh caption cho ảnh

## 1. Nhận dạng đề

- Mỗi ảnh train có vài câu mô tả (thường 5); với mỗi ảnh test viết **một** câu. Chung kết 2025 có bài này. Ngôn ngữ có thể là tiếng Anh (ImgCap: COCO / nocaps) hoặc tiếng Việt (KTVIC_Captioning tự host).
- Luật thường gặp: phần ảnh được dùng backbone ImageNet trong danh sách, **phần ngôn ngữ train từ đầu**, không CLIP / BLIP.

## 2. Metric

CIDEr-D (chính), BLEU-4, ROUGE-L, đôi khi METEOR / SPICE. Xem [01_metric.md](../01_metric.md): CIDEr thưởng n-gram **đặc trưng** của ảnh, phạt chênh độ dài so với tham chiếu.

## 3. Lộ trình

| Bước | Việc | Kỳ vọng |
|---|---|---|
| 0 | Baseline BTC, nộp | = Min |
| 1 | **Truy hồi + đồng thuận**: đặc trưng ảnh ImageNet (GAP ResNet-50 / ConvNeXt-T, TTA lật), K ảnh train gần nhất, chọn câu đồng thuận nhất trong 5K câu (MBR theo CIDEr) | thường đã mạnh với dữ liệu ít (vài nghìn ảnh) |
| 2 | Chọn K, trọng số theo độ giống ảnh, IDF trên caption train | ImgCap: tăng K tới 30-50 có lợi |
| 3 | Sinh câu: đặc trưng ảnh (đóng băng) + decoder LSTM / Transformer nhỏ train từ đầu, beam search 3-5 | tốt khi câu rập khuôn (KTVIC) và đủ dữ liệu |
| 4 | Sinh nhiều câu (beam + truy hồi) rồi chọn bằng MBR | kết hợp ưu điểm |

## 4. Khung code: truy hồi + MBR

```python
def cider_vec(caps, df, log_n):
    """TF-IDF n-gram 1-4 cho mỗi câu (đã tách từ). df: tần suất văn bản của n-gram trên caption train (theo ảnh)."""
    out = []
    for t in caps:
        c = Counter(tuple(t[i:i + k]) for k in range(1, 5) for i in range(len(t) - k + 1))
        v = {g: tf * (log_n - math.log(max(1.0, df[g]))) for g, tf in c.items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        out.append({g: x / n for g, x in v.items()})
    return out

def mbr(ung_vien, trong_so, df, log_n):
    """Chọn câu có tổng tương đồng (cosine TF-IDF) có trọng số lớn nhất với các ứng viên khác."""
    V = cider_vec([chuan_hoa(c) for c in ung_vien], df, log_n)
    diem = [sum(w * sum(a.get(g, 0) * x for g, x in b.items()) for b, w, j in zip(V, trong_so, range(len(V))) if j != i)
            for i, a in enumerate(V)]
    return ung_vien[int(np.argmax(diem))]

S = Fq @ Ftrain.T                                   # đặc trưng đã chuẩn hoá L2
for i, s in enumerate(S):
    nb = np.argsort(-s)[:K]
    ung = [c for j in nb for c in caps[j]]; w = [s[j] for j in nb for _ in caps[j]]
    pred[i] = mbr(ung, w, df, log_n)
```

## 5. Khung code: decoder sinh câu

```python
class CapDec(nn.Module):
    def __init__(self, V, D_img, d=512, L=3):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Embedding(40, d); self.img = nn.Linear(D_img, d)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, 8, 4 * d, 0.2, batch_first=True), L)
        self.out = nn.Linear(d, V)
    def forward(self, f_img, tok):                    # f_img: (B, P, D_img) đặc trưng lưới (vd 7x7=49 vị trí); tok: (B, L)
        mem = self.img(f_img)
        x = self.emb(tok) + self.pos(torch.arange(tok.shape[1], device=tok.device))
        mask = nn.Transformer.generate_square_subsequent_mask(tok.shape[1], device=tok.device)
        return self.out(self.dec(x, mem, tgt_mask=mask, tgt_key_padding_mask=tok == PAD))
```

- Train: teacher forcing, mỗi epoch chọn ngẫu nhiên 1 trong 5 câu của ảnh, `cross_entropy(..., ignore_index=PAD, label_smoothing=0.1)`.
- Đặc trưng lưới: bỏ `avgpool` và `fc` của ResNet, lấy (B, 2048, 7, 7) rồi `flatten(2).transpose(1, 2)`; cache trước.
- Giải mã beam 3-5, chặn lặp n-gram (không cho lặp lại trigram đã có), độ dài tối thiểu.
- Vocab: từ xuất hiện ≥ 3-5 lần; tiếng Việt tách theo âm tiết là đủ.

## 6. Nút vặn

1. Đặc trưng ảnh tốt hơn (backbone mạnh hơn trong danh sách, ảnh lớn hơn, TTA lật).
2. K và trọng số trong MBR; IDF trên caption train.
3. Độ dài câu: CIDEr phạt chênh độ dài, chọn câu dài gần trung bình tham chiếu.
4. Decoder: label smoothing, dropout, beam + chặn lặp; **trung bình checkpoint**.
5. Kết hợp ứng viên từ truy hồi và từ decoder rồi MBR.

## 7. Bẫy

- Dữ liệu ít (vài nghìn ảnh): decoder train từ đầu sinh câu chung chung, lặp từ; truy hồi thường thắng.
- Test khác miền train (nocaps có vật thể mới): truy hồi trả câu về vật thể quen thuộc, điểm thấp; không có cách sửa nhanh, chấp nhận.
- Chuẩn hoá Unicode tiếng Việt (NFC) khi tách từ và tính IDF.

## 8. Bài luyện

- Code tay: [d02_caption](../../luyen_tap/CV/d02_caption/) (truy hồi + MBR, decoder nhỏ, CIDEr tự viết).
- Thi thử: [KTVIC_Captioning](../../selfhost_kaggle/KTVIC_Captioning/) (tiếng Việt; baseline màu thô 0,196 / 0,217).
- Lời giải tham khảo: [challenges/ImgCap](../../challenges/ImgCap/) (ResNet-18 từ đầu học "từ khái niệm" + truy hồi + MBR).

## 9. Đọc thêm

- Vedantam và cộng sự, *CIDEr* ([arXiv:1411.5726](https://arxiv.org/abs/1411.5726)).
- Devlin và cộng sự, *Exploring Nearest Neighbor Approaches for Image Captioning* ([arXiv:1505.04467](https://arxiv.org/abs/1505.04467)): truy hồi + đồng thuận.
- Xu và cộng sự, *Show, Attend and Tell* ([arXiv:1502.03044](https://arxiv.org/abs/1502.03044)).
