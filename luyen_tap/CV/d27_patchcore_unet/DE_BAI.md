# Bài 27 (27/11) - PatchCore + U-Net encoder ResNet

**Thời gian mục tiêu: ≤ 50 phút** (A khoảng 20, B khoảng 30).

## Đọc trước khi làm

Đọc trong khoảng 30-45 phút trước khi bấm giờ; ghi ra giấy 3-5 ý chính (công thức, tham số, bẫy). Lúc làm bài thì đóng hết.

- **Bài báo**:
  - PatchCore: Roth và cộng sự [arXiv:2106.08265](https://arxiv.org/abs/2106.08265), mục 3.1-3.3 (đặc trưng tầng giữa, local aggregation, coreset, điểm ảnh = max khoảng cách patch).
  - PaDiM [arXiv:2011.08785](https://arxiv.org/abs/2011.08785), mục 3 (đọc để so: mô hình Gauss theo vị trí thay cho kNN).
  - U-Net: Ronneberger và cộng sự [arXiv:1505.04597](https://arxiv.org/abs/1505.04597), mục 2 (kiến trúc encoder-decoder, skip connection).
  - Dice loss: Milletari và cộng sự, V-Net [arXiv:1606.04797](https://arxiv.org/abs/1606.04797), mục 3 (công thức Dice mềm).
- **Bộ dữ liệu tham khảo**: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) (xem ảnh mẫu để hiểu bài toán).
- **PyTorch / torchvision**: [torch.cdist](https://pytorch.org/docs/stable/generated/torch.cdist.html), [Feature extraction](https://pytorch.org/vision/stable/feature_extraction.html), [F.interpolate](https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html).
- **Trong máy**: `../../ON_TAP.md` phần 1.8 (phát hiện bất thường); `../../../challenges/CCTV_Street_Scence/solution.ipynb` (kNN patch, luôn kiểm tra chiều của điểm).

## Phần A - Phát hiện ảnh lỗi chỉ từ ảnh bình thường (`bai_a.py`)

- **Đầu vào**: `data/a_train/` 200 ảnh bề mặt vân sọc **bình thường**; `data/a_test/` 200 ảnh (một nửa có lỗi nhẹ: vết xước, đốm, mảng mất vân).
- **Đầu ra**: `submission_a.csv` (`id,score`), điểm càng cao càng bất thường.
- **Metric**: AUROC, **ngưỡng 0.78**.
- **Khung**: ResNet-18 ImageNet, lấy `layer2` + `layer3` (`create_feature_extractor`), phóng về cùng kích thước, nối kênh, làm mượt 3x3; memory bank từ patch ảnh train (rút gọn ngẫu nhiên khoảng 16k); điểm ảnh = **max theo patch** của khoảng cách tới láng giềng gần nhất (`torch.cdist`).

## Phần B - Phân đoạn vùng lỗi (`bai_b.py`)

- **Đầu vào**: `data/b_train/images/` + `data/b_train/masks/` (400 ảnh lỗi, mặt nạ 0/255); `data/b_test/` 100 ảnh.
- **Đầu ra**: `masks_pred/<id>.png` (64x64, 0/255).
- **Metric**: Dice trung bình theo ảnh, **ngưỡng 0.80**.
- **Khung**: U-Net với **encoder ResNet-18 ImageNet** (stem, layer1-4), decoder: phóng to + `torch.cat` skip + conv; loss BCE + Dice; lật ngẫu nhiên **cùng lúc ảnh và mặt nạ**; val Dice sau mỗi epoch, lưu checkpoint tốt nhất; ngưỡng 0.5.

## Bài 2 - Viết lại nhanh bài 26 (≤ 30 phút)

Trong `d26_video`: `viet_lai.py`, chấm `python check_26.py viet_lai.py`.

## Cách chạy

```
python tao_du_lieu.py
python check_27.py bai_a.py bai_b.py
```

Lời giải mẫu: AUROC 0.84, Dice 0.93, tổng khoảng 1.5 phút.
