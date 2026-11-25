# Bài 25 (25/11) - Tấn công đối kháng + Grad-CAM

**Thời gian mục tiêu: ≤ 40 phút.**

## Đọc trước khi làm

Đọc trong khoảng 30-45 phút trước khi bấm giờ; ghi ra giấy 3-5 ý chính (công thức, tham số, bẫy). Lúc làm bài thì đóng hết.

- **PyTorch**: [Adversarial Example Generation](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html) (FGSM từng bước), [torch.autograd.grad](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html).
- **Bài báo**:
  - FGSM: Goodfellow và cộng sự [arXiv:1412.6572](https://arxiv.org/abs/1412.6572), mục 4 (công thức một bước theo dấu gradient).
  - PGD: Madry và cộng sự [arXiv:1706.06083](https://arxiv.org/abs/1706.06083), mục 2 (bước lặp + chiếu về quả cầu, khởi đầu ngẫu nhiên).
  - Grad-CAM: Selvaraju và cộng sự [arXiv:1610.02391](https://arxiv.org/abs/1610.02391), mục 3 (trọng số kênh = trung bình gradient, ReLU).
  - Pointing game: Zhang và cộng sự, *Top-down Neural Attention by Excitation Backprop* [arXiv:1608.00507](https://arxiv.org/abs/1608.00507), mục 5.1.
- **PSNR**: Wikipedia [Peak signal-to-noise ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) (công thức theo MSE; lệch đều ±8 cho khoảng 30.1 dB).
- **Trong máy**: `../../ON_TAP.md` phần 1.9 (đề chung kết 2025 có tấn công đối kháng).

## Bài 1 - FGSM / PGD và lưu ảnh đối kháng

- **Đầu vào**: `mo_hinh.py` + `mo_hinh.pt` (CNN đã train, **không sửa**; đầu vào là ảnh float trong [0, 1], tự chuẩn hoá bên trong); `data/test/*.png` (100 ảnh mô hình đoán đúng), `data/test.csv`, `data/classes.csv` (thứ tự lớp).
- **Đầu ra**: `adv/<id>.png` cho cả 100 ảnh.
- **Ràng buộc và metric**:
  - lệch tối đa so với ảnh gốc **≤ 8** (tức eps = 8/255, chuẩn L_inf) **sau khi lưu PNG uint8**;
  - PSNR trung bình **≥ 30 dB**;
  - tỉ lệ tấn công thành công (dự đoán sai) khi đọc lại PNG **≥ 0.55**. FGSM một bước chỉ được khoảng 0.42; PGD đúng được khoảng 0.62.
- **Cần làm**: FGSM; PGD (khởi đầu ngẫu nhiên, bước alpha = 2/255, 10 bước, **chiếu về quả cầu L_inf** rồi về [0, 1]); làm tròn sang uint8 mà vẫn giữ |lệch| ≤ 8; **đọc lại PNG để kiểm tra** (PSNR, tỉ lệ thành công).

## Bài 2 - Grad-CAM

- **Đầu ra**: `cam.npy` shape `(100, 64, 64)`, theo thứ tự `data/test.csv`, giá trị không âm.
- **Metric**: pointing game, điểm nóng nhất của CAM nằm trong vật thể (cho sai số 4 px) ở **≥ 75%** ảnh.
- **Cần làm**: lấy bản đồ đặc trưng cuối (`m.features`), gradient của logit lớp đúng theo bản đồ đó, trọng số kênh = trung bình gradient, `ReLU(tổng có trọng số)`, phóng về 64x64.

## Bài 3 - Viết lại nhanh bài 24 (≤ 30 phút)

Trong `d24_truy_xuat`: `viet_lai.py`, chấm `python check_24.py viet_lai.py`.

## Cách chạy

```
python tao_du_lieu.py              # train sẵn mo_hinh.pt (khoảng 1.5 phút trên GTX 1650)
python check_25.py bai_lam.py
```
