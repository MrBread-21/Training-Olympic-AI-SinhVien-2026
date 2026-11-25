# Dạng 6 - Tấn công đối kháng và độ bền mô hình

## 1. Nhận dạng đề

- **Tấn công**: cho mô hình (và ảnh), tạo ảnh lệch rất ít (giới hạn L∞ hoặc PSNR) để mô hình đoán sai. Chung kết 2025: tạo ảnh đối kháng để thử độ bền mô hình nhận diện hoạt động. Có thể có mô hình ẩn (cần chuyển giao), lớp đích (targeted), hoặc điểm phụ theo chất lượng ảnh (PSNR, SSIM).
- **Phòng thủ**: train mô hình vẫn đúng trên ảnh đối kháng / ảnh hỏng.

## 2. Metric

Tỉ lệ đánh lừa thành công (với ảnh hợp lệ), đôi khi nhân / cộng điểm chất lượng ảnh. Đọc kỹ: ảnh vi phạm giới hạn thường được **0**, nên kẹp và kiểm tra trước khi nộp.

## 3. Lộ trình (tấn công)

| Bước | Việc | Kỳ vọng (CIFAR_Adversarial tự host) |
|---|---|---|
| 0 | FGSM một bước với ε tối đa, nộp | lừa A 98%, điểm 0,81 |
| 1 | PGD 10-20 bước (bước α = ε/4), khởi tạo ngẫu nhiên | lừa A gần 100%; chuyển giao có khi giảm |
| 2 | Tăng chuyển giao: MI-FGSM (quán tính), DI (biến đổi đầu vào ngẫu nhiên), tấn công ensemble mô hình thay thế tự train | điểm mô hình ẩn tăng |
| 3 | Lượng tử hoá đúng: làm tròn, kẹp `[x-ε, x+ε]` và `[0, 255]` **sau** khi làm tròn; lưu PNG (không JPEG) | tránh ảnh không hợp lệ |

## 4. Khung code

```python
def pgd(models, x, y, eps=8/255, alpha=2/255, steps=10, mu=1.0, di=True, rand_init=True):
    """x: (B,3,H,W) 0-1. models: list mô hình (ensemble thay thế). mu > 0: MI-FGSM. di: đa dạng đầu vào."""
    adv = x + (torch.empty_like(x).uniform_(-eps, eps) if rand_init else 0)
    adv, g = adv.clamp(0, 1).detach(), torch.zeros_like(x)
    for _ in range(steps):
        adv.requires_grad_(True)
        inp = adv
        if di and random.random() < 0.7:                     # DI: phóng nhỏ ngẫu nhiên rồi đệm về kích thước cũ
            s = random.randint(int(0.85 * x.shape[-1]), x.shape[-1])
            inp = F.interpolate(adv, size=(s, s), mode="bilinear", align_corners=False)
            p = x.shape[-1] - s; l = random.randint(0, p); t = random.randint(0, p)
            inp = F.pad(inp, (l, p - l, t, p - t))
        loss = sum(F.cross_entropy(m(inp), y) for m in models)
        grad, = torch.autograd.grad(loss, adv)
        g = mu * g + grad / grad.abs().mean((1, 2, 3), keepdim=True)    # MI-FGSM
        adv = adv.detach() + alpha * g.sign()
        adv = torch.min(torch.max(adv, x - eps), x + eps).clamp(0, 1)
    return adv.detach()

def luong_tu(adv, x_uint8, eps_int=8):
    """Về uint8 hợp lệ: làm tròn rồi kẹp lại theo ảnh gốc."""
    a = (adv * 255).round().to(torch.int16); x = x_uint8.to(torch.int16)
    return torch.min(torch.max(a, x - eps_int), x + eps_int).clamp(0, 255).to(torch.uint8)
```

Kiểm tra trước khi nộp: `(adv_uint8.int() - x_uint8.int()).abs().max() <= eps_int`, và tỉ lệ đánh lừa trên mô hình công bố **sau khi lượng tử hoá** (làm tròn có thể làm mất hiệu lực tấn công ở biên).

Targeted: `loss = -F.cross_entropy(m(inp), y_dich)` (đi về phía lớp đích).

## 5. Phòng thủ (khi đề yêu cầu mô hình bền)

- Adversarial training: mỗi batch tạo PGD 3-7 bước trên mô hình hiện tại, train trên ảnh đối kháng (hoặc trộn 50/50 với ảnh sạch).
- Augmentation mạnh (nhiễu, JPEG, mờ), TTA; đầu vào làm mượt nhẹ (median filter) có thể giảm hiệu lực nhiễu tần số cao.

## 6. Nút vặn (tấn công chuyển giao)

1. Ensemble nhiều mô hình thay thế khác seed / khác kiến trúc (train nhanh trên dữ liệu được cấp).
2. MI + DI; thêm dịch chuyển (TI) hoặc trộn ảnh khác lớp (Admix).
3. Số bước vừa phải (10-20); quá nhiều bước khớp sát mô hình thay thế.
4. Tối đa loss **trên logit** (CW margin) thay vì cross-entropy khi mô hình quá tự tin.

## 7. Bẫy

- Chuẩn hoá mean/std nằm trong mô hình hay ngoài: tấn công trong không gian 0-1 của ảnh, ε tính trên ảnh gốc.
- `model.eval()` (BatchNorm) và tắt dropout khi tính gradient tấn công.
- Lưu JPEG phá hỏng nhiễu đối kháng; luôn PNG / mảng thô.

## 8. Bài luyện

- Code tay: [d25_doi_khang](../../luyen_tap/CV/d25_doi_khang/) (FGSM, PGD, PSNR, Grad-CAM).
- Thi thử: [CIFAR_Adversarial](../../selfhost_kaggle/CIFAR_Adversarial/) (mô hình A công bố, B ẩn; baseline FGSM 0,809).

## 9. Đọc thêm

- Goodfellow và cộng sự, *Explaining and Harnessing Adversarial Examples* (FGSM, [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)).
- Madry và cộng sự, *Towards Deep Learning Models Resistant to Adversarial Attacks* (PGD, [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)).
- Dong và cộng sự, *Boosting Adversarial Attacks with Momentum* (MI-FGSM, [arXiv:1710.06081](https://arxiv.org/abs/1710.06081)); Xie và cộng sự, *Improving Transferability with Input Diversity* (DI, [arXiv:1803.06978](https://arxiv.org/abs/1803.06978)).
