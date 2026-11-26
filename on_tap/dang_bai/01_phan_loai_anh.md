# Dạng 1 - Phân loại ảnh (kể cả ảnh test bị hỏng, nhãn nhiễu, ít nhãn)

## 1. Nhận dạng đề

- Đầu vào: thư mục ảnh + `train.csv` (`id,label`). Đầu ra: `id,label` cho mỗi ảnh test.
- Biến thể hay gặp trong đề Olympic: ảnh test **bị làm hỏng** (xoay, JPEG, mờ, nhiễu, tối), nhãn train **nhiễu** một phần, **ít nhãn** + nhiều ảnh không nhãn, lớp mất cân bằng, ảnh nhỏ (32-64 px) hoặc rất lớn.
- Đã gặp: EuroSAT (ảnh vệ tinh, test hỏng, 1.000 ảnh có nhãn), "nhận diện hoạt động từ ảnh" (chung kết 2025), FGVC_Aircraft_Noisy (tự host: 70 lớp, 15% nhãn sai, 60% ảnh test hỏng).

## 2. Metric

Accuracy hoặc macro-F1 (xem [01_metric.md](../01_metric.md)). Macro-F1: chú ý lớp hiếm, chỉnh hệ số từng lớp trên OOF.

## 3. Lộ trình trong 6 tiếng

| Bước | Việc | Thời gian | Kỳ vọng |
|---|---|---|---|
| 0 | Chạy baseline BTC, nộp | 15 phút | = Min |
| 1 | EDA: số ảnh / lớp, kích thước, xem 30 ảnh train và 30 ảnh test cạnh nhau (test hỏng kiểu gì?) | 10 phút | |
| 2 | **Đặc trưng đóng băng** từ backbone ImageNet được phép (ResNet-50 / ConvNeXt-Tiny, ảnh 224, TTA lật) + LogisticRegression, CV 5-fold | 20 phút | vượt baseline rõ |
| 3 | Fine-tune backbone với augmentation mô phỏng test, val bị làm hỏng giống test | 45-90 phút | tăng tiếp |
| 4 | TTA, ensemble 2-3 mô hình (khác backbone hoặc khác seed), pseudo-label ảnh không nhãn | còn lại | |

Nếu đề **cấm pretrained**: bỏ bước 2, ResNet-18 `weights=None` (ảnh ≤ 64 px thì `conv1` 3x3 stride 1, bỏ maxpool), train 50-200 epoch, augmentation mạnh.

## 4. Khung code

Nạp ảnh nhỏ lên GPU, augmentation bằng tensor (nhanh hơn DataLoader nhiều lần):

```python
X = torch.tensor(np.stack([np.asarray(Image.open(p).convert("RGB").resize((S, S))) for p in paths])).permute(0, 3, 1, 2).contiguous()  # uint8 N,3,S,S
X, Y = X.to(DEV), torch.tensor(labels, device=DEV)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

def aug(xb):                       # xb float 0-1
    if random.random() < 0.5: xb = xb.flip(3)
    k = random.randint(0, 3); xb = torch.rot90(xb, k, (2, 3))          # ảnh vệ tinh: nhóm D4
    xb = xb * (0.7 + 0.6 * torch.rand(len(xb), 1, 1, 1, device=DEV))  # sáng tối
    xb = xb + 0.05 * torch.randn_like(xb) * (torch.rand(len(xb), 1, 1, 1, device=DEV) < 0.3)
    return xb.clamp(0, 1).contiguous()

for ep in range(EP):
    for idx in torch.randperm(len(X), device=DEV).split(BS):
        xb = (aug(X[idx].float() / 255) - MEAN) / STD
        loss = F.cross_entropy(model(xb), Y[idx], label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
```

Backbone ImageNet được phép, thay head:

```python
m = torchvision.models.resnet50(weights="DEFAULT"); m.fc = nn.Linear(m.fc.in_features, C)
m = torchvision.models.convnext_tiny(weights="DEFAULT"); m.classifier[2] = nn.Linear(768, C)
m = torchvision.models.efficientnet_b0(weights="DEFAULT"); m.classifier[1] = nn.Linear(1280, C)
# trích đặc trưng: m.fc = nn.Identity() (ResNet) / m.classifier = nn.Flatten() (ConvNeXt: thêm LayerNorm có sẵn ở classifier[0])
```

JPEG / mờ trên GPU (mô phỏng ảnh hỏng): JPEG làm sẵn một bản hỏng của train ở CPU bằng PIL (`img.save(buf, "JPEG", quality=q)`) rồi trộn với bản gốc; mờ bằng `torchvision.transforms.functional.gaussian_blur(xb, 5, sigma)`.

TTA D4:

```python
@torch.inference_mode()
def predict_tta(model, X):
    out = 0
    for k in range(4):
        for f in (False, True):
            x = torch.rot90(X, k, (2, 3)); x = x.flip(3) if f else x
            out = out + model(x).softmax(1)
    return out / 8
```

## 5. Nút vặn (thứ tự đáng thử)

1. Augmentation **đúng kiểu test** + val làm hỏng giống test.
2. Độ phân giải: phóng ảnh nhỏ lên 2-4 lần khi dùng backbone ImageNet (64 lên 224).
3. Train đủ lâu, OneCycle, label smoothing; EMA trọng số.
4. TTA; ensemble 2-3 mô hình.
5. Pseudo-label dữ liệu không nhãn (ngưỡng 0,9, cân bằng lớp).
6. Nhãn nhiễu: small-loss trick (sau vài epoch bỏ 10-20% mẫu loss lớn nhất), label smoothing 0,2.
7. Macro-F1: hệ số lớp trên OOF.
8. Bản cuối train trên train + val.

## 6. Bẫy

- Chọn mô hình trên val sạch khi test hỏng (EuroSAT: val sạch 0,895, LB 0,70).
- Quên `model.eval()` khi dự đoán (BatchNorm, dropout).
- Đọc ảnh bằng `cv2` ra BGR rồi dùng mean/std ImageNet của RGB.
- Ảnh xám / RGBA lẫn trong dữ liệu: luôn `.convert("RGB")`.
- Đặt tên lớp sai thứ tự khi ghi file nộp: giữ `classes = sorted(...)` một chỗ duy nhất.

## 7. Bài luyện

- Code tay: [d21_phan_loai_anh](../../luyen_tap/CV/d21_phan_loai_anh/) (khung fine-tune), [d22_augment_nhan_nhieu](../../luyen_tap/CV/d22_augment_nhan_nhieu/) (ảnh hỏng, MixUp, small-loss, TTA), [d23_dac_trung_dong_bang](../../luyen_tap/CV/d23_dac_trung_dong_bang/) (đặc trưng đóng băng, CV theo nhóm).
- Thi thử: [FGVC_Aircraft_Noisy](../../selfhost_kaggle/FGVC_Aircraft_Noisy/) (baseline 0,338).
- Lời giải tham khảo: [challenges/EuroSAT](../../challenges/EuroSAT/) (không pretrained, augmentation mô phỏng test, pseudo-label, TTA D4).

## 8. Đọc thêm

- He và cộng sự, *Bag of Tricks for Image Classification* ([arXiv:1812.01187](https://arxiv.org/abs/1812.01187)).
- Hendrycks và Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions* ([arXiv:1903.12261](https://arxiv.org/abs/1903.12261)): các kiểu làm hỏng ảnh.
- Han và cộng sự, *Co-teaching* ([arXiv:1804.06872](https://arxiv.org/abs/1804.06872)): học với nhãn nhiễu.
