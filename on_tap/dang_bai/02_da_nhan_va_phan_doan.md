# Dạng 2 - Phân loại đa nhãn và phân đoạn ảnh

## A. Phân loại đa nhãn

### Nhận dạng đề

Mỗi ảnh có **0, 1 hoặc nhiều** nhãn (`labels` là các tên lớp cách nhau dấu cách). Ví dụ: VOC_MultiLabel (tự host, 20 lớp), gắn thẻ ảnh, phát hiện nhiều loại bệnh.

### Cách làm

- Đầu ra C logit, loss `BCEWithLogitsLoss` (không softmax). Nhãn mã hoá multi-hot.
- Lớp hiếm: `pos_weight` trong BCE, hoặc ASL / focal loss.
- **Ngưỡng từng lớp** chọn trên OOF để tối đa F1 của lớp đó (quan trọng hơn đổi mô hình):

```python
def chon_nguong(P, Y):                      # P, Y: (N, C) xác suất và nhãn 0/1
    t = np.zeros(P.shape[1])
    for c in range(P.shape[1]):
        cand = np.linspace(0.05, 0.95, 19)
        f = [2 * ((P[:, c] >= x) & (Y[:, c] == 1)).sum() / max(1, (P[:, c] >= x).sum() + Y[:, c].sum()) for x in cand]
        t[c] = cand[int(np.argmax(f))]
    return t
```

- Ảnh không có nhãn nào vượt ngưỡng: tuỳ đề, có thể gán lớp xác suất cao nhất (nếu mỗi ảnh chắc chắn có ít nhất một nhãn).

### Bài luyện

[VOC_MultiLabel](../../selfhost_kaggle/VOC_MultiLabel/) (baseline 0,714).

## B. Phân đoạn ảnh (segmentation)

### Nhận dạng đề

Đầu ra là mask từng điểm ảnh, nộp dạng RLE. Ví dụ: Kvasir_PolypSegmentation (tự host), phân đoạn tổn thương y tế, vùng hư hỏng sản phẩm.

### Metric

Dice / IoU trung bình theo ảnh ([01_metric.md](../01_metric.md)). Ảnh không có vật: dự đoán rỗng mới được điểm.

### Lộ trình

| Bước | Việc | Kỳ vọng |
|---|---|---|
| 0 | Baseline BTC (thường ngưỡng màu / U-Net nhỏ), nộp; viết và **kiểm tra RLE encode / decode** trước | = Min |
| 1 | U-Net encoder ResNet-18/34 ImageNet (nếu được), ảnh 256-352, BCE + Dice, lật / xoay / màu | vượt baseline nhiều |
| 2 | TTA lật, chọn ngưỡng trên val, bỏ vùng nhỏ | +0,01-0,03 |
| 3 | Ảnh lớn hơn, ensemble 2 encoder | |

### Khung code: U-Net với encoder ResNet

```python
class UNetRes(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet34(weights="DEFAULT" if pretrained else None)
        self.l0 = nn.Sequential(r.conv1, r.bn1, r.relu)            # 1/2, 64
        self.l1 = nn.Sequential(r.maxpool, r.layer1)              # 1/4, 64
        self.l2, self.l3, self.l4 = r.layer2, r.layer3, r.layer4  # 1/8 128, 1/16 256, 1/32 512
        def up(ci, cs, co):
            return nn.ModuleDict({"u": nn.ConvTranspose2d(ci, co, 2, 2),
                                  "c": nn.Sequential(nn.Conv2d(co + cs, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True),
                                                     nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True))})
        self.d4, self.d3, self.d2, self.d1 = up(512, 256, 256), up(256, 128, 128), up(128, 64, 64), up(64, 64, 64)
        self.out = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, 2), nn.ReLU(True), nn.Conv2d(32, 1, 1))

    def forward(self, x):
        e0 = self.l0(x); e1 = self.l1(e0); e2 = self.l2(e1); e3 = self.l3(e2); e4 = self.l4(e3)
        d = e4
        for blk, skip in ((self.d4, e3), (self.d3, e2), (self.d2, e1), (self.d1, e0)):
            d = blk["u"](d); d = blk["c"](torch.cat([d, skip], 1))
        return self.out(d)                                        # logit cùng kích thước ảnh vào (ảnh chia hết cho 32)

def bce_dice(logit, y):
    p = logit.sigmoid(); inter = (p * y).sum((2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((2, 3)) + y.sum((2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logit, y) + dice.mean()
```

RLE (duyệt theo hàng, bắt đầu từ 1; **đọc kỹ đề** vì có đề duyệt theo cột):

```python
def rle_encode(m):
    p = np.concatenate([[0], m.flatten(), [0]]); r = np.where(p[1:] != p[:-1])[0] + 1; r[1::2] -= r[::2]
    return " ".join(map(str, r))
def rle_decode(s, h, w):
    m = np.zeros(h * w, np.uint8); a = list(map(int, s.split()))
    for st, ln in zip(a[0::2], a[1::2]): m[st - 1:st - 1 + ln] = 1
    return m.reshape(h, w)
```

Augmentation hình học phải áp **cùng lúc** cho ảnh và mask (lật, xoay 90 độ, crop); màu chỉ áp cho ảnh. Dự đoán ở kích thước làm việc rồi `F.interpolate` về kích thước gốc **trước** khi lấy ngưỡng.

### Nút vặn

1. Encoder pretrained (nếu được) và ảnh đủ lớn.
2. BCE + Dice; Lovasz hinge khi tối ưu IoU.
3. TTA lật ngang / dọc, trung bình xác suất.
4. Ngưỡng và kích thước vùng tối thiểu chọn trên val.
5. Hậu xử lý: giữ thành phần liên thông lớn nhất nếu mỗi ảnh chỉ có một vật (`scipy.ndimage.label`).

### Bẫy

- Mask lưu dạng màu (VOC: bảng màu), dạng 0/255, hoặc có viền 255 "bỏ qua": kiểm tra `np.unique(mask)` trước.
- Nội suy mask bằng bilinear tạo giá trị lẻ: resize mask dùng `NEAREST`.
- RLE sai thứ tự duyệt cho điểm gần 0 dù mô hình tốt: giải mã lại và so với mask val trước khi nộp.

### Bài luyện

- Code tay: [d27_patchcore_unet](../../luyen_tap/CV/d27_patchcore_unet/) phần B (U-Net encoder ResNet-18, BCE + Dice).
- Thi thử: [Kvasir_PolypSegmentation](../../selfhost_kaggle/Kvasir_PolypSegmentation/) (baseline 0,508).

### Đọc thêm

- Ronneberger và cộng sự, *U-Net* ([arXiv:1505.04597](https://arxiv.org/abs/1505.04597)).
- Berman và cộng sự, *The Lovász-Softmax loss* ([arXiv:1705.08790](https://arxiv.org/abs/1705.08790)).
