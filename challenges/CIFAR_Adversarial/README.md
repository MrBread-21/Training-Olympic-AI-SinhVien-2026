# Lời giải CIFAR_Adversarial - Tấn công đối kháng CIFAR-10

Đề: [selfhost_kaggle/CIFAR_Adversarial](../../selfhost_kaggle/CIFAR_Adversarial/). Code: [solve.py](solve.py). Không dùng model pretrained.

## Kết quả (chấm bằng đúng metric của cuộc thi, `selfhost_kaggle/tools/cham.py`)

| | Public | Private |
|---|---|---|
| Baseline của đề | 0,809 | 0,808 (FGSM) |
| **Lời giải này** | 0,962 | 0,963 |

Thời gian chạy: 11 phút trên GTX 1650 (phần lớn là train 3 mô hình thay thế).

## Cách làm

1. Train **3 mô hình thay thế** từ đầu trên `cifar_train.npz` (CNN 3 khối, rộng 32 / 32 / 48, seed khác nhau, 15 epoch, SGD OneCycle, lật + dịch ảnh).
2. Tấn công **ensemble** (A + 3 mô hình thay thế): tổng cross-entropy của 4 mô hình.
3. **PGD 20 bước** (bước 1,6/255) với **MI-FGSM** (cộng dồn gradient chuẩn hoá, mu = 1) và **DI** (70% số bước: thu nhỏ ngẫu nhiên 27-32 px rồi đệm về 32).
4. Lượng tử hoá: làm tròn về uint8 rồi **kẹp lại** trong [gốc - 8, gốc + 8] và [0, 255] sau khi làm tròn.

## Ghi chú

Kết quả: lừa A khoảng 98%, lừa mô hình ẩn B khoảng 94% (suy ra từ điểm). FGSM một bước chỉ chuyển giao được khoảng 64%. Ensemble mô hình thay thế + MI + DI là phần quyết định; tấn công riêng A bằng PGD nhiều bước thường chuyển giao kém hơn.

## Chạy lại

```
python solve.py <thư_mục upload của đề>     # giải nén private_test.zip vào thư mục làm việc trước nếu muốn dự đoán cả private
python ../../selfhost_kaggle/tools/cham.py CIFAR_Adversarial submission.csv
```
