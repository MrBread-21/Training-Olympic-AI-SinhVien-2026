# Bài 22 (22/11) - Augmentation mạnh + nhãn nhiễu

**Thời gian mục tiêu: ≤ 45 phút.**

## Đọc trước khi làm

Đọc trong khoảng 30-45 phút trước khi bấm giờ; ghi ra giấy 3-5 ý chính (công thức, tham số, bẫy). Lúc làm bài thì đóng hết.

- **torchvision transforms v2**: [Getting started](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html) và [How to use CutMix and MixUp](https://pytorch.org/vision/stable/auto_examples/transforms/plot_cutmix_mixup.html) (áp dụng sau DataLoader, nhãn mềm).
- **Bài báo** (đọc phần phương pháp, 1-2 trang mỗi bài):
  - RandAugment [arXiv:1909.13719](https://arxiv.org/abs/1909.13719), mục 3 (hai tham số N, M).
  - mixup [arXiv:1710.09412](https://arxiv.org/abs/1710.09412), mục 2; CutMix [arXiv:1905.04899](https://arxiv.org/abs/1905.04899), mục 3.1.
  - Label smoothing: Müller và cộng sự, *When Does Label Smoothing Help?* [arXiv:1906.02629](https://arxiv.org/abs/1906.02629), mục 1-2.
  - Nhãn nhiễu: Arpit và cộng sự [arXiv:1706.05394](https://arxiv.org/abs/1706.05394) (mạng học mẫu sạch trước, học thuộc nhiễu sau); Co-teaching [arXiv:1804.06872](https://arxiv.org/abs/1804.06872), mục 3 (chọn mẫu loss nhỏ).
  - He và cộng sự, *Bag of Tricks for Image Classification* [arXiv:1812.01187](https://arxiv.org/abs/1812.01187), mục 5 (label smoothing, mixup, cosine lr).
- **Trong máy**: `../../EuroSAT/train_img.py` (augment mô phỏng test trên GPU, val bị làm hỏng, pseudo-label, TTA D4).

## Bài 1 - Phân loại khi nhãn sai và ảnh test bị làm hỏng

- **Đầu vào**: 1800 ảnh train sạch nhưng **khoảng 1/3 nhãn bị gán sai** (nhiễu đối xứng); 600 ảnh test bị **xoay 90/lật, mờ, nhiễu, lệch màu, JPEG chất lượng thấp**. Cấu trúc thư mục giống bài 21.
- **Đầu ra**: `submission.csv` (`id,label`).
- **Metric**: macro-F1, **ngưỡng 0.90**. Khung bài 21 để nguyên chỉ được khoảng 0.82.
- **Kỹ thuật cần dùng**:
  1. Augmentation mô phỏng test: xoay 90 / lật, `v2.RandAugment`, Gaussian blur, `v2.JPEG`, nhiễu Gauss.
  2. MixUp / CutMix (`v2.MixUp`, `v2.CutMix`, nhãn mềm) và `label_smoothing`.
  3. **Small-loss**: train giai đoạn 1 trên toàn bộ, tính loss từng mẫu (không augment), giữ khoảng 60% mẫu loss nhỏ nhất **trong từng lớp**, train lại giai đoạn 2.
  4. TTA 8 phép (4 góc xoay x có/không lật), trung bình xác suất.
- Không có val sạch: nhãn val cũng nhiễu. Hãy nghĩ xem nên chọn số epoch thế nào.

## Bài 2 - Viết lại nhanh bài 21 (≤ 25 phút)

Trong thư mục `d21_phan_loai_anh`, viết `viet_lai.py` từ trí nhớ, rồi `python check_21.py viet_lai.py`. Đạt tất cả mục là xong.

## Cách chạy

```
python tao_du_lieu.py
python check_22.py bai_lam.py
```

Lời giải mẫu (`loi_giai/giai.py`) đạt khoảng 0.98, chạy khoảng 1 phút trên GTX 1650.
