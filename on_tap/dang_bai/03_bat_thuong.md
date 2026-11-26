# Dạng 3 - Phát hiện bất thường (ảnh sản phẩm, camera giám sát)

## 1. Nhận dạng đề

- Train **chỉ có mẫu bình thường** (hoặc rất ít mẫu bất thường), test lẫn cả hai. Đầu ra: điểm bất thường cho mỗi ảnh / khung (`id,score`), đôi khi kèm bản đồ vùng lỗi.
- Đã gặp: CCTV Street Scene (khung video camera cố định), MVTec_AnomalyDetection (tự host, 5 loại sản phẩm), MSAD (bất thường trong video có nhãn loại sự kiện, xem [dạng 5](05_video.md)).

## 2. Metric

AUROC (thường) hoặc AP. Chỉ thứ tự điểm có ý nghĩa. **Luôn kiểm tra chiều**: nếu val dưới 0,5 thì đổi dấu.

## 3. Lộ trình

| Bước | Việc | Kỳ vọng |
|---|---|---|
| 0 | Baseline BTC, nộp | = Min |
| 1 | Đặc trưng CNN ImageNet (layer2 + layer3 của ResNet / WideResNet) + **kNN tới tập đặc trưng bình thường** (PatchCore rút gọn) | thường vượt xa baseline với ảnh sản phẩm |
| 2 | Chuẩn hoá điểm theo loại sản phẩm / theo camera nếu metric gộp nhiều nhóm | |
| 3 | Kết hợp nhiều thang (ảnh to nhỏ), nhiều đặc trưng, rank average | |

Không được pretrained: autoencoder nhỏ (lỗi tái tạo), thống kê pixel theo vị trí (trung vị, MAD), PCA trên patch.

## 4. Khung code: PatchCore rút gọn

```python
r = torchvision.models.resnet50(weights="DEFAULT")   # trong danh sách BTC (bài báo gốc dùng wide_resnet50_2, không có trong danh sách)
r.eval().to(DEV)                                     # resnet50: layer2 512 kênh + layer3 1024 kênh, C = 1536
feat = {}
r.layer2.register_forward_hook(lambda m, i, o: feat.__setitem__("l2", o))
r.layer3.register_forward_hook(lambda m, i, o: feat.__setitem__("l3", o))

@torch.inference_mode()
def patch_feats(x):                                        # x: (B,3,224,224) đã chuẩn hoá
    r(x)
    l2 = F.avg_pool2d(feat["l2"], 3, 1, 1)                 # làm mượt lân cận 3x3
    l3 = F.interpolate(F.avg_pool2d(feat["l3"], 3, 1, 1), size=l2.shape[-2:], mode="bilinear", align_corners=False)
    f = torch.cat([l2, l3], 1)                             # (B, C, 28, 28)
    return f.permute(0, 2, 3, 1).reshape(len(x), -1, f.shape[1])   # (B, 784, C)

bank = torch.cat([patch_feats(xb).reshape(-1, C) for xb in normal_batches])   # mọi patch bình thường
bank = bank[torch.randperm(len(bank))[:50000]]             # lấy mẫu con (coreset ngẫu nhiên) cho vừa bộ nhớ

@torch.inference_mode()
def score(xb):
    f = patch_feats(xb)                                    # (B, P, C)
    d = torch.stack([torch.cdist(fi, bank).min(1).values for fi in f])   # (B, P) khoảng cách tới patch bình thường gần nhất
    return d.max(1).values                                 # điểm ảnh = patch lạ nhất; d.reshape(B,28,28) là bản đồ vùng lỗi
```

## 5. Nút vặn

1. Chọn tầng đặc trưng (layer2 + layer3 cân bằng giữa chi tiết và ngữ nghĩa).
2. Ảnh lớn hơn với lỗi nhỏ (xước, vết); crop trung tâm nếu nền thừa.
3. Điểm ảnh: max, hoặc trung bình top-k patch (ổn định hơn với nhiễu).
4. Một memory bank **mỗi loại sản phẩm / mỗi camera**; chuẩn hoá điểm theo nhóm nếu metric gộp nhóm và tỉ lệ bất thường các nhóm giống nhau.
5. Video: làm mượt điểm theo thời gian (trung bình trượt), kết hợp đặc trưng chuyển động (hiệu khung).

## 6. Bẫy (bài học từ CCTV)

- Detector "khác bình thường" có thể **đo nhầm thứ khác** (mật độ xe, ánh sáng): CCTV mọi detector kNN chạy ngược chiều (val 0,25-0,30); đổi dấu điểm "khác nền trung vị" cho val 0,743.
- AUROC gộp nhiều video: tỉ lệ bất thường giữa các video chênh lớn (5% và 83%) thì phần lớn tín hiệu là mức video; chuẩn hoá theo video kéo AUROC về 0,5. Tách điểm thành `z(trong video) + a x z(mức video)`.
- Val ít video: không tin chênh lệch nhỏ, dùng lượt nộp public để chọn hướng.

## 7. Bài luyện

- Code tay: [d27_patchcore_unet](../../luyen_tap/CV/d27_patchcore_unet/) phần A (PatchCore).
- Thi thử: [MVTec_AnomalyDetection](../../selfhost_kaggle/MVTec_AnomalyDetection/) (baseline 0,642).
- Lời giải tham khảo: [challenges/CCTV_Street_Scence](../../challenges/CCTV_Street_Scence/) (đọc kỹ phần EDA).

## 8. Đọc thêm

- Roth và cộng sự, *Towards Total Recall in Industrial Anomaly Detection* (PatchCore, [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)).
- Defard và cộng sự, *PaDiM* ([arXiv:2011.08785](https://arxiv.org/abs/2011.08785)): Gauss từng vị trí patch.
