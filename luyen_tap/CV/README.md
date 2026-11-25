# Luyện code tay - chung kết OlpAI'26 (20/11 đến 5/12)

Mỗi ngày một khung code cần thuộc, luyện theo điều kiện phòng thi: **đóng tài liệu, không LLM, bấm giờ**. Mỗi bài có dữ liệu giả lập tạo bằng script (chạy CPU dưới 2 phút), bộ chấm tự động, và lời giải mẫu tối đa 80 dòng để đối chiếu sau khi làm.

## Cách luyện mỗi ngày

1. Đọc `dNN_.../DE_BAI.md`, chạy `python tao_du_lieu.py` (chỉ lần đầu).
2. Bấm giờ, viết `bai_lam.py` trong thư mục bài. Không mở `loi_giai/` và `.dap_an/`.
3. `python check_NN.py bai_lam.py`: bộ chấm kiểm tra quy tắc code, chạy bài làm, kiểm tra định dạng file nộp và điểm so với ngưỡng. Thêm `--khong-chay` nếu chỉ muốn chấm file nộp đã có.
4. Đạt hết thì dừng giờ, ghi vào bảng tiến độ cuối file. Chưa đạt thì sửa tiếp, ghi chú lỗi đã mắc.
5. Mở `loi_giai/` so cách viết; chép lại 3 điểm muốn nhớ.
6. Làm bài "viết lại nhanh" khung của ngày trước (ghi trong `DE_BAI.md`), chấm bằng bộ chấm của ngày đó.

## Lịch

| Ngày | Thư mục | Khung | Mục tiêu | Ngưỡng chấm | Lời giải mẫu |
|---|---|---|---|---|---|
| 20/11 | `d20_tinh_toan` | numpy: chuẩn hoá, softmax, cross-entropy, accuracy, ma trận nhầm lẫn, macro-F1, AUROC, cosine, Euclid, kNN, làm mượt, tIoU, Jaccard, Kadane | ≤ 60 phút | 16/16 hàm đúng | 16/16 |
| 21/11 | `d21_phan_loai_anh` | Dataset/DataLoader, fine-tune ResNet-18, val ngoài vòng batch, lưu best, predict theo batch, nộp theo id | ≤ 45 phút | macro-F1 ≥ 0.95 | 0.98 |
| 22/11 | `d22_augment_nhan_nhieu` | RandAugment, JPEG/blur/nhiễu, MixUp/CutMix, label smoothing, small-loss, TTA 8 phép | ≤ 45 phút | macro-F1 ≥ 0.90 (khung 21 để nguyên: 0.82) | 0.99 |
| 23/11 | `d23_dac_trung_dong_bang` | ResNet-50 đóng băng + cache, logistic / kNN, StratifiedGroupKFold, OOF, ma trận nhầm lẫn | ≤ 30 phút | macro-F1 ≥ 0.85 | 0.90 |
| 24/11 | `d24_truy_xuat` | GeM, ArcFace, kNN cosine, Recall@K, val theo đối tượng | ≤ 45 phút | Recall@1 ≥ 0.75 (GAP đóng băng: 0.22) | 0.79-0.83 |
| 25/11 | `d25_doi_khang` | FGSM, PGD L_inf, lưu PNG + PSNR, Grad-CAM | ≤ 40 phút | thành công ≥ 0.55, PSNR ≥ 30, pointing ≥ 0.75 | 0.61 / 31.9 dB / 0.85 |
| 26/11 | `d26_video` | N khung rải đều, cache đặc trưng, gộp thời gian + hiệu t/t+k, logistic, điểm khung + đoạn tIoU kỳ vọng | ≤ 50 phút | điểm macro ≥ 0.50 | 0.61 |
| 27/11 | `d27_patchcore_unet` | PatchCore (layer2+3, memory bank, kNN); U-Net encoder ResNet-18, BCE + Dice | ≤ 50 phút | AUROC ≥ 0.78, Dice ≥ 0.80 | 0.84 / 0.93 |
| 28/11 | `d28_nlp` | NFC, tokenize, vocab, bucketing; TF-IDF từ + ký tự + logistic; TextCNN; BM25 + Recall@1 | ≤ 50 phút | F1 ≥ 0.87 / 0.85, R@1 ≥ 0.85 (quên NFC: 0.80) | 0.91 / 0.90 / 0.93 |
| 29/11 | `d29_seq2seq` | nn.Transformer từ đầu, mặt nạ, teacher forcing, greedy + beam, BLEU / chrF | ≤ 50 phút | BLEU ≥ 0.90 | 0.99 |
| 30/11 | `d30_du_lieu_bang` | ColumnTransformer, encoding, HistGradientBoosting, OOF, stacking | ≤ 35 phút | AUROC ≥ 0.73 (logistic một mình: 0.70) | 0.75 |
| 1/12 | `d01_am_thanh` | wav, log-mel tự tính bằng `torch.stft`, CNN 2D, trộn tạp âm theo SNR, SpecAugment, val theo người nói | ≤ 50 phút | macro-F1 ≥ 0.85 (không trộn tạp âm: 0.35) | 0.92 |
| 2/12 | `d02_caption` | A: CNN học "từ khái niệm" + truy hồi K láng giềng + MBR; B: encoder CNN + decoder GRU, teacher forcing, greedy | ≤ 70 phút | CIDEr-D ≥ 3.6 / 4.5 (truy hồi bằng pixel: 2.63) | 4.27 / 4.82 |
| 3/12 | `d03_hoi_dap` | NFC, chọn câu, sinh span, đặc trưng span, XGBoost xếp hạng, val theo đoạn văn, ngưỡng để trống | ≤ 60 phút | F1 ≥ 0.80 (quên NFC: 0.60) | 0.88 |
| 4/12 | `d04_dimabsa` | A: mệnh đề chứa khía cạnh + TF-IDF + Ridge (RMSE_VA); B: BIO BiLSTM + ký tự, ghép cặp, category, VA (cF1) | ≤ 75 phút | RMSE_VA ≤ 0.70, cF1 ≥ 0.70 (tra từ điển: 0.57) | 0.50 / 0.79 |
| 5/12 | `d05_duong_di` | A: Dijkstra + Held-Karp; B: value iteration trên gridworld trơn trượt | ≤ 45 phút | 0.99 / 0.97 (BFS tham lam 0.79; bỏ qua trượt 0.93) | 1.00 / 1.00 |

Mục tiêu thời gian là thời gian **viết + chạy** đến khi bộ chấm báo đạt. Lời giải mẫu chạy trên GTX 1650 4 GB (fp32) mỗi bài dưới 3 phút (d02 khoảng 4 phút mỗi phần).

Bài học lý thuyết đi kèm từng ngày nằm ở [`on_tap/dang_bai/`](../../on_tap/dang_bai/); lộ trình tổng ở [`on_tap/README.md`](../../on_tap/README.md).

## Quy tắc trong mọi bài làm

Bộ chấm tự kiểm tra các mục đánh dấu (tự động); các mục còn lại tự soát.

- `seed_all(42)` (random, numpy, torch, cuda) - **tự động**
- bf16 / autocast chỉ khi `torch.cuda.get_device_capability()[0] >= 8`, còn lại fp32 - **tự động**
- Dự đoán theo batch trong `model.eval()` + `torch.inference_mode()` - **tự động**
- Optimizer và scheduler tạo **một lần**, ngoài vòng lặp - **tự động** (báo đúng dòng sai)
- Validation **ngoài vòng batch** - **tự động** (cảnh báo)
- Đổi thứ tự trục bằng `permute` / `transpose`, không dùng `reshape` - **tự động** (liệt kê các dòng có reshape/view để tự soát)
- Chuẩn hoá `unicodedata.normalize("NFC", ...)` với văn bản - **tự động** ở bài NLP
- Điền file nộp **theo id**, đúng tên file và tên cột, không thiếu id, không NaN - **tự động**
- Nhãn `dtype=torch.long`; `num_workers=0` trên Windows; `import torch` trước `cv2` / `pandas` (lỗi DLL c10.dll trên Windows)

## Bài học đã rút ra (áp dụng khi làm)

1. Làm baseline đơn giản trước: đặc trưng đóng băng + logistic thường thắng mô hình phức tạp khi ít dữ liệu.
2. Val nhỏ thì dùng CV; dữ liệu có nhóm thì chia fold theo nhóm (bài 23 cho thấy OOF chia ngẫu nhiên cao hơn thật 0.06).
3. Bản cuối train trên toàn bộ dữ liệu có nhãn.
4. Đọc ma trận nhầm lẫn trước khi đổi mô hình.

## Môi trường

- **Máy thi**: Linux, torch 2.8 + cu128, torchvision 0.23, numpy 2.x, transformers 5.22 nhưng **không có pretrained NLP**; chỉ backbone ImageNet của torchvision (resnet18/34/50, convnext_tiny, efficientnet_b0, mobilenet_v2/v3, regnet_y_400mf, densenet121...).
- **Máy nhà**: Windows, GTX 1650 4 GB, fp32. Trọng số ImageNet được tải về lần đầu, các lần sau dùng cache.

## Cấu trúc

```
luyen_tap/CV/
├── chung/cham.py          bộ chấm dùng chung (quy tắc code, chạy bài làm, đọc file nộp)
├── chung/hinh.py          sinh ảnh hình học giả lập cho các bài CV
├── chung/chay_tat_ca.py   chấm lời giải mẫu của mọi ngày (dùng khi sửa bộ chấm)
└── dNN_<chủ đề>/
    ├── DE_BAI.md          đề + bài viết lại nhanh
    ├── tao_du_lieu.py     tạo data/ và .dap_an/ (đáp án, không mở)
    ├── check_NN.py        bộ chấm
    └── loi_giai/          lời giải mẫu (chỉ mở sau khi làm xong)
```

## Bảng tiến độ

| Ngày | Bài | Thời gian | Đạt / trượt | Điểm | Lỗi đã mắc | Ghi nhớ |
|---|---|---|---|---|---|---|
| | | | | | | |
