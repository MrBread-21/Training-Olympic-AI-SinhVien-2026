# Bài 23 (23/11) - Đặc trưng đóng băng + CV theo nhóm + OOF

**Thời gian mục tiêu: ≤ 30 phút.**

## Đọc trước khi làm

Đọc trong khoảng 30-45 phút trước khi bấm giờ; ghi ra giấy 3-5 ý chính (công thức, tham số, bẫy). Lúc làm bài thì đóng hết.

- **scikit-learn**: [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), các mục *GroupKFold*, *StratifiedGroupKFold*, *cross_val_predict*; [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) (rò rỉ dữ liệu); [Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline) (`make_pipeline(StandardScaler(), LogisticRegression())`).
- **torchvision**: [Feature extraction](https://pytorch.org/vision/stable/feature_extraction.html) (`create_feature_extractor`, hoặc thay `fc` bằng `nn.Identity()`).
- **Bài báo**: Kornblith và cộng sự, *Do Better ImageNet Models Transfer Better?* [arXiv:1805.08974](https://arxiv.org/abs/1805.08974), mục 3-4 (logistic trên đặc trưng đóng băng là baseline mạnh).
- **Trong máy**: `../../ON_TAP.md` phần 2.E; [README bài CCTV](https://github.com/MrBread-21/Training-Olympic-AI-SinhVien-2026/tree/MrBread/CCTV_Street_Scence) trên GitHub (bài học "val ít nhóm không đáng tin"), `../../MSAD/main.py` (LR trên đặc trưng ResNet đóng băng, CV theo nhóm cảnh).

## Bài 1 - Baseline nhanh và đánh giá đúng

- **Đầu vào**: `data/train.csv` (`id,group,label`): 768 ảnh, chia thành 96 **nhóm** ("phiên chụp" 8 ảnh cùng nền, cùng màu, cùng cỡ vật thể). `data/test/`: 288 ảnh thuộc 36 nhóm **mới**.
- **Đầu ra**: `submission.csv` (`id,label`) và `oof.csv` (`id,label`: dự đoán ngoài fold cho mọi ảnh train).
- **Metric**: macro-F1 test, **ngưỡng 0.85**. Bộ chấm cảnh báo nếu OOF cao hơn test quá 0.04 (dấu hiệu rò rỉ).
- **Khung bắt buộc**:
  1. Backbone ImageNet **đóng băng** (ResNet-50 hoặc ResNet-18, bỏ `fc`), trích đặc trưng một lần, **lưu cache** `.npy`.
  2. So sánh LogisticRegression (có StandardScaler) với kNN cosine.
  3. **`StratifiedGroupKFold` theo cột `group`**, dự đoán OOF, in macro-F1 OOF và **ma trận nhầm lẫn**; đọc xem lớp nào hay nhầm với lớp nào.
  4. Bản cuối: fit lại trên toàn bộ train, dự đoán test.
- Tự thử: thay bằng `StratifiedKFold` thường rồi so OOF. Lời giải mẫu: OOF theo nhóm 0.885, OOF chia ngẫu nhiên 0.956, test 0.899.

## Bài 2 - Viết lại nhanh bài 22 (≤ 30 phút)

Trong `d22_augment_nhan_nhieu`, viết `viet_lai.py` (augment + MixUp + small-loss + TTA), chấm bằng `python check_22.py viet_lai.py`.

## Cách chạy

```
python tao_du_lieu.py
python check_23.py bai_lam.py
```
