# EuroSAT - Phân loại ảnh vệ tinh khi ít nhãn

Bài luyện tập IMGCLS trên [hệ thống ôn luyện PHENIKAA AI Hackathon](https://aihackathon.phenikaa-uni.edu.vn/).

## Bài toán

- Phân loại ảnh Sentinel-2 RGB 64x64 vào 10 lớp phủ đất (`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`).
- Dữ liệu: 1000 ảnh train có nhãn (100 ảnh/lớp), 500 ảnh val, 3000 ảnh không nhãn, 4000 ảnh test.
- Ảnh test **bị làm hỏng**: xoay/lật, nhiễu Gauss mạnh, mờ, lệch màu, nén JPEG chất lượng rất thấp. Ảnh train và val thì sạch.
- Điểm: macro-F1 (phụ: accuracy).

**LB public tốt nhất: 0.8074.** Điểm này khớp với bản fine-tune ResNet-50 pretrained ở mục "Phiên bản trước" (val hard 0.793); hai bản không pretrained trong repo không có ghi chép LB riêng.

Cả hai lời giải dưới đây **không dùng trọng số pretrained** (chỉ dùng kiến trúc, trọng số tự train trên dữ liệu được cấp), theo luật của kỳ thi OlpAI mà đội đang luyện.

## Ý tưởng chung

1. **Augmentation mô phỏng đúng biến đổi của test**: 8 phép xoay/lật (nhóm D4), jitter màu, blur, nhiễu Gauss, nén JPEG chất lượng thấp.
2. **Chọn model trên val bị làm hỏng** ("val hard": mỗi ảnh val làm hỏng một lần với seed cố định), không chọn trên val sạch. Val sạch đánh giá quá cao: ở phiên bản dùng pretrained trước đó, val sạch 0.895 trong khi LB chỉ 0.7015; val hard 0.683 sát LB hơn nhiều.
3. **TTA 8 phép D4** khi dự đoán, vì test bị xoay/lật ngẫu nhiên.
4. **Pseudo-label** ảnh `unlabeled` (không pseudo-label ảnh test).

## Bản 1: `solution.ipynb` (3 tầng)

| Tầng | Cách làm | val sạch | val hard |
|---|---|---|---|
| 1 | Đặc trưng thủ công bất biến khi xoay (thống kê màu RGB/HSV, histogram, LBP `uniform` P=8/16, GLCM trung bình 4 hướng, gradient), train trên ảnh gốc và 3 bản làm hỏng; so sánh LogisticRegression với XGBoost, chọn theo val hard (LR) | 0.798 | 0.596 |
| 2a | ResNet-18 khởi tạo ngẫu nhiên, conv đầu đổi thành 3x3 stride 1 cho ảnh 64x64; AdamW lr 2e-3, OneCycle, label smoothing 0.1; số epoch tự tính theo ngân sách 12 phút (71 epoch trên GTX 1650); TTA 8 phép | 0.773 | 0.654 |
| 2a + 1 | Cộng xác suất tầng 1 vào tầng 2a (chọn trên val hard) | - | **0.665** |
| 2b | Pseudo-label ảnh unlabeled (xác suất ≥ 0.85, tối đa 300 ảnh/lớp, giữ được 946/3000), train lại trên train + val + pseudo trong 12 phút, ensemble với 2a và tầng 1 | - | - |

Số liệu lấy từ log lần chạy trên GTX 1650, fp32. Cả notebook chạy khoảng 30 phút.

## Bản 2: `train_img.py` (bản cuối)

Viết lại gọn hơn, dữ liệu và augmentation đặt hết trên GPU (`olp_core.py`):

- **Augmentation trên GPU**: D4 (lật ngang, lật dọc, chuyển vị), random-resized-crop dạng affine, lệch màu và độ bão hoà, blur 5x5 hai mức, nhiễu Gauss. Nén JPEG làm sẵn trên CPU: mỗi ảnh có thêm 2 bản JPEG chất lượng ngẫu nhiên 10-60, lấy mẫu cân bằng giữa bản sạch và bản nén.
- **Model**: ResNet-18 từ đầu (conv đầu 3x3 stride 1, giữ maxpool), AdamW lr 2e-3 weight decay 0.05 (không decay bias và BatchNorm), OneCycle, nhãn mềm (0.9 + 0.01), **EMA** trọng số 0.999.
- **Giai đoạn 1**: 15 epoch trên ảnh có nhãn; lưu checkpoint EMA tốt nhất theo macro-F1 trên val bị làm hỏng.
- **Giai đoạn 2**: pseudo-label ảnh unlabeled có xác suất (TTA 8 phép) > 0.8, chỉ chạy nếu giữ được ≥ 300 ảnh; train lại 10 epoch.
- **Dự đoán**: TTA 8 phép; ghép model giai đoạn 1 nếu nó không kém giai đoạn 2 quá 0.02 macro-F1 trên val bị làm hỏng.
- Có các bước kiểm tra trước khi ghi file nộp: không có NaN, ảnh test cùng định dạng với train, không lớp nào chiếm hơn 50% dự đoán.

Lần chạy đầy đủ của bản này không lưu lại log điểm.

## Phiên bản trước (dùng pretrained, code không còn giữ)

Trước khi chuyển sang luật không dùng pretrained, đội đã thử ResNet-50 ImageNet:

| Cách làm | val sạch | val hard | LB |
|---|---|---|---|
| ResNet-50 đóng băng + LogisticRegression (C=0.05) trên đặc trưng của ảnh train và 3 bản làm hỏng, TTA 2 phép | 0.895 | 0.683 | 0.7015 |
| Fine-tune ResNet-50, ảnh 128px, 19 epoch, TTA 8 phép | 0.942 | 0.793 | **0.8074** |

## Bài học

- **fp16 trên GPU GTX 16xx ra NaN** tùy batch size, phải chạy fp32. bf16 chỉ bật trên GPU Ampere trở lên.
- ConvNeXt chạy rất chậm trên GTX 16xx (conv depthwise 7x7 và LayerNorm), ResNet nhanh hơn hẳn.
- Ghép mô hình yếu không phải lúc nào cũng tăng điểm: với bản pretrained, ghép tầng 1 vào fine-tune làm val hard giảm (0.783 so với 0.793). Vì vậy notebook chọn việc ghép dựa trên val hard.

## Chạy

Đặt dữ liệu của bài (`train/`, `val/`, `unlabeled/`, `test/`, `train_labels.csv`, `val_labels.csv`, `classes.csv`, `sample_submission.csv`) cùng thư mục với code, rồi:

```
jupyter nbconvert --to notebook --execute solution.ipynb   # bản 1
python train_img.py                                         # bản 2
```

Thư viện: `torch`, `torchvision`, `numpy`, `pandas`, `opencv-python`, `Pillow`, `scikit-image`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `tqdm`.

## Tham khảo

- Helber, P. và cộng sự (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*. IEEE JSTARS. [arXiv:1709.00029](https://arxiv.org/abs/1709.00029).
- He, K. và cộng sự (2016). *Deep Residual Learning for Image Recognition*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
- Ojala, T., Pietikäinen, M., Mäenpää, T. (2002). *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns*. IEEE TPAMI.
- Haralick, R. M., Shanmugam, K., Dinstein, I. (1973). *Textural Features for Image Classification*. IEEE TSMC (GLCM).
- Lee, D.-H. (2013). *Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks*. ICML Workshop.
- Smith, L. N., Topin, N. (2017). *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates* (OneCycle). [arXiv:1708.07120](https://arxiv.org/abs/1708.07120).
