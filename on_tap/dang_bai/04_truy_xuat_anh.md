# Dạng 4 - Truy xuất ảnh, nhận dạng địa điểm (VPR)

## 1. Nhận dạng đề

- Có **truy vấn** (query) và **kho** (database). Với mỗi truy vấn, trả về ảnh kho khớp nhất (hoặc K ảnh xếp hạng).
- Đã gặp: Nordland (tác vụ 2 vòng sơ loại OlpAI'26: ảnh mùa hè bị làm hỏng tìm ảnh mùa đông cùng vị trí, đúng nếu lệch ≤ 10 khung, Recall@1 thang 0-100); tìm sản phẩm giống nhau; nhận dạng lại người / xe.

## 2. Metric

Recall@1 (hoặc Recall@K, mAP, MRR). Xem [01_metric.md](../01_metric.md).

## 3. Lộ trình

| Bước | Việc | Kỳ vọng (Nordland) |
|---|---|---|
| 0 | Baseline BTC (pixel thu nhỏ + cosine), nộp | Recall@1 khoảng 4-14 |
| 1 | Đặc trưng ImageNet dùng nguyên (GAP) + cosine | **có thể còn tệ hơn** (Nordland khoảng 3): đặc trưng phân loại không bất biến mùa |
| 2 | Train đặc trưng bằng **contrastive** giữa cặp (truy vấn, ảnh kho đúng) của train, giữ **bố cục không gian** (lưới từ layer3) | Nordland bản cũ: val R@1 0,93 sau 25 phút |
| 3 | Augmentation mô phỏng làm hỏng truy vấn (tối, mờ, sương, JPEG, độ phân giải thấp, mưa, chói), re-ranking | |

## 4. Khung code: descriptor lưới + InfoNCE với negative khó

```python
class LuoiDesc(nn.Module):
    def __init__(self, luoi=(4, 7), d=64):
        super().__init__()
        r = torchvision.models.resnet18(weights="DEFAULT")
        self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)   # 256 kênh
        self.proj = nn.Conv2d(256, d, 1); self.luoi = luoi
    def forward(self, x):
        f = F.adaptive_avg_pool2d(self.proj(self.body(x)), self.luoi)      # (B, d, 4, 7): giữ vị trí trái / phải / trên / dưới
        f = F.normalize(f, dim=1).flatten(1)
        return F.normalize(f, dim=1)

def info_nce(q, k, tau=0.05):                  # q[i] và k[i] là cùng vị trí; các k[j] khác là negative
    logits = q @ k.t() / tau
    t = torch.arange(len(q), device=q.device)
    return (F.cross_entropy(logits, t) + F.cross_entropy(logits.t(), t)) / 2
```

Batch quyết định độ khó: lấy một **đoạn liên tiếp** các vị trí (ví dụ 64 vị trí cách nhau 3-5 khung) làm một batch, nên negative là các cảnh gần giống nhau (negative khó). Đóng băng BatchNorm của backbone (`m.eval()` cho các lớp BN khi train) vì batch có cấu trúc đặc biệt; không đóng băng thì mô hình sụp.

Dự đoán:

```python
Dq, Ddb = embed(queries), embed(database)          # (Nq, D), (Ndb, D) đã chuẩn hoá
best = (Dq @ Ddb.t()).argmax(1)                    # nhiều truy vấn: chia khối để vừa bộ nhớ
```

## 5. Nút vặn

1. Giữ không gian (lưới, NetVLAD / GeM theo vùng) thay vì GAP toàn ảnh.
2. Negative khó (khung liền kề) + nhiệt độ tau 0,03-0,1.
3. Augmentation phía truy vấn giống kiểu làm hỏng của test; phía kho để sạch.
4. TTA lật ngang **không** dùng (đổi trái phải làm sai bố cục đường ray); dùng đa thang (ảnh to nhỏ).
5. Re-ranking: trung bình descriptor truy vấn với vài láng giềng gần nhất (query expansion); với kho liên tiếp theo thời gian, làm mượt điểm tương đồng dọc trục kho.
6. Bản cuối train trên train + validation.

## 6. Bẫy

- Kho có nhiều ảnh gần như giống nhau (khung liền kề): đề chấm theo dung sai vị trí nên chọn ảnh nào trong vùng đúng cũng được; đừng phạt chúng như negative ở bước train nếu nằm trong dung sai.
- Tên file ngẫu nhiên: không suy ra vị trí từ tên hay thứ tự.
- Bộ nhớ: `Nq x Ndb` ma trận tương đồng có thể rất lớn, chia khối 1.000 truy vấn.

## 7. Bài luyện

- Code tay: [d24_truy_xuat](../../luyen_tap/CV/d24_truy_xuat/) (GeM, ArcFace, kNN cosine, Recall@K).
- Thi thử: [Nordland_PlaceRecognition](../../selfhost_kaggle/Nordland_PlaceRecognition/) (đúng định dạng đề sơ loại; baseline 4,22 / 13,70).
- Lời giải tham khảo: [challenges/Nordland_PlaceRecognition](../../challenges/Nordland_PlaceRecognition/) (viết cho bản đề giả định cũ nhưng kỹ thuật dùng lại được); repo T-Sunm/Nordland-Cross-Season-VPR (Bag-of-Queries, Multi-Similarity loss).

## 8. Đọc thêm

- Arandjelović và cộng sự, *NetVLAD* ([arXiv:1511.07247](https://arxiv.org/abs/1511.07247)).
- Radenović và cộng sự, *Fine-tuning CNN Image Retrieval with No Human Annotation* (GeM, [arXiv:1711.02512](https://arxiv.org/abs/1711.02512)).
- Ali-bey và cộng sự, *BoQ: A Place is Worth a Bag of Learnable Queries* ([arXiv:2405.07364](https://arxiv.org/abs/2405.07364)).
