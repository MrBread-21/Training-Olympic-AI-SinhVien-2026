# Nhánh MrBread

Bài làm của MrBread trong quá trình chuẩn bị cho chung kết Olympic AI sinh viên 2026 (OlpAI'26), gồm hai phần:

1. [`luyen_tay/`](luyen_tay/): bộ bài luyện code tay theo từng ngày (20/11 đến 30/11).
2. Lời giải 5 bài trên hệ thống ôn luyện PHENIKAA AI Hackathon (EuroSAT, ImgCap, VietnameseExtract, CCTV_Street_Scence, MSAD).

## Bộ bài luyện code tay (`luyen_tay/`)

Mỗi ngày một khung code cần thuộc, luyện theo điều kiện phòng thi: đóng tài liệu, không LLM, bấm giờ. Mỗi bài gồm:

- `DE_BAI.md`: đề như đề thi (đầu vào, đầu ra, metric, ngưỡng, thời gian mục tiêu), danh sách tài liệu nên đọc trước, và một bài "viết lại nhanh" khung của ngày trước.
- `tao_du_lieu.py`: tạo dữ liệu giả lập (chạy CPU dưới 2 phút) và đáp án ẩn.
- `check_NN.py`: bộ chấm tự động: kiểm tra quy tắc code (seed 42, `model.eval()` + `inference_mode`, bf16 chỉ khi GPU hỗ trợ, optimizer tạo ngoài vòng lặp, validation ngoài vòng batch, chuẩn hoá NFC), chạy bài làm, kiểm tra file nộp và so điểm với ngưỡng.
- `loi_giai/`: lời giải mẫu tối đa 80 dòng, chỉ mở sau khi làm xong.

| Ngày | Thư mục | Khung | Ngưỡng chấm | Lời giải mẫu |
|---|---|---|---|---|
| 20/11 | `d20_tinh_toan` | numpy: chuẩn hoá, softmax, cross-entropy, macro-F1, AUROC, kNN, tIoU, Kadane (16 hàm) | 16/16 | 16/16 |
| 21/11 | `d21_phan_loai_anh` | Dataset/DataLoader, fine-tune ResNet-18, lưu best, predict theo batch, nộp theo id | macro-F1 ≥ 0.95 | 0.98 |
| 22/11 | `d22_augment_nhan_nhieu` | RandAugment, JPEG/blur/nhiễu, MixUp/CutMix, label smoothing, small-loss, TTA | macro-F1 ≥ 0.90 | 0.98 |
| 23/11 | `d23_dac_trung_dong_bang` | Đặc trưng đóng băng + cache, logistic / kNN, StratifiedGroupKFold, OOF | macro-F1 ≥ 0.85 | 0.90 |
| 24/11 | `d24_truy_xuat` | GeM, ArcFace, kNN cosine, Recall@K | Recall@1 ≥ 0.75 | 0.79-0.83 |
| 25/11 | `d25_doi_khang` | FGSM, PGD, PSNR sau khi lưu PNG, Grad-CAM | thành công ≥ 0.55, PSNR ≥ 30, pointing ≥ 0.75 | 0.61 / 31.9 dB / 0.85 |
| 26/11 | `d26_video` | N khung rải đều, cache, gộp thời gian, định vị đoạn theo tIoU kỳ vọng | điểm macro ≥ 0.50 | 0.61 |
| 27/11 | `d27_patchcore_unet` | PatchCore (kNN patch); U-Net encoder ResNet-18, BCE + Dice | AUROC ≥ 0.78, Dice ≥ 0.80 | 0.84 / 0.94 |
| 28/11 | `d28_nlp` | NFC, vocab, bucketing; TF-IDF + logistic; TextCNN; BM25 | F1 ≥ 0.87 / 0.85, Recall@1 ≥ 0.85 | 0.91 / 0.90 / 0.93 |
| 29/11 | `d29_seq2seq` | nn.Transformer từ đầu, mặt nạ, greedy + beam, BLEU / chrF | BLEU ≥ 0.90 | 0.99 |
| 30/11 | `d30_du_lieu_bang` | ColumnTransformer, encoding, HistGradientBoosting, stacking OOF | AUROC ≥ 0.73 | 0.75 |

Ngưỡng được đặt để tách cách làm đúng khỏi cách làm sơ sài, ví dụ: giữ nguyên khung bài 21 cho bài 22 chỉ được 0.82, quên chuẩn hoá NFC ở bài 28 chỉ được 0.80, chỉ dùng FGSM ở bài 25 chỉ được 0.42.

Cách dùng (xem thêm [`luyen_tay/README.md`](luyen_tay/README.md)):

```
cd luyen_tay/d21_phan_loai_anh
python tao_du_lieu.py              # tạo data/ và .dap_an/ (không đưa lên repo)
python check_21.py bai_lam.py      # chạy bài làm rồi chấm
```

Một số đề nhắc tới `ON_TAP.md`: đây là ghi chú ôn tập cá nhân, không có trong repo.

## Lời giải các bài ôn luyện PHENIKAA AI Hackathon

Lời giải của MrBread cho các bài trên [hệ thống ôn luyện PHENIKAA AI Hackathon](https://aihackathon.phenikaa-uni.edu.vn/), trong quá trình chuẩn bị cho kỳ thi Olympic AI sinh viên 2026.

Phần lớn lời giải được viết theo luật của kỳ thi: **không dùng trọng số pretrained** (chỉ dùng kiến trúc, trọng số tự train trên dữ liệu được cấp), hoặc chỉ dùng backbone ImageNet trong danh sách được cấp và để đóng băng. Hầu hết chạy được trên GPU 4 GB (GTX 1650, fp32) trong khoảng 20 phút.

| Bài | Nhiệm vụ | Hướng giải chính | LB public tốt nhất |
|---|---|---|---|
| [EuroSAT](EuroSAT/) | Phân loại ảnh vệ tinh 10 lớp khi ít nhãn, ảnh test bị làm hỏng | Augmentation mô phỏng biến đổi của test, chọn model trên val bị làm hỏng, TTA 8 phép xoay/lật; bản trong repo dùng ResNet-18 train từ đầu + pseudo-label | **0.8074** (bản fine-tune ResNet-50 pretrained); bản trong repo: val hard 0.665 |
| [ImgCap](ImgCap/) | Mô tả ảnh (COCO, test nocaps) | Bản trong repo: ResNet-18 train từ đầu dự đoán 400 từ khái niệm, truy hồi 20 ảnh gần nhất, chọn câu đồng thuận (MBR theo CIDEr) | **1.1927** (bản BLIP pretrained); bản trong repo: val CIDEr-D 0.2725 |
| [VietnameseExtract](VietnameseExtract/) | Hỏi đáp trích xuất tiếng Việt | Bản trong repo: chọn câu theo IDF, sinh span ứng viên, xếp hạng bằng XGBRanker, thêm SQuAD tiếng Anh | **0.6517** (bản XLM-R pretrained); bản trong repo: val Jaccard 0.227 |
| [CCTV_Street_Scence](CCTV_Street_Scence/) | Điểm bất thường từng khung, camera giám sát cố định | Điểm mức video (đường vắng) + kNN của blob vật thể so với video bình thường, chạy CPU, không pretrained | **0.6761**; `solve.py` mặc định: 0.6727 |
| [MSAD](MSAD/) | Phân loại và định vị sự kiện bất thường trong video | Đặc trưng chuyển động/màu thủ công; mô hình bảng + CNN train từ đầu, hoặc backbone ImageNet đóng băng + LR + TCN | **0.4149** (`main_scratch.py`, không pretrained) |

Với EuroSAT, ImgCap và VietnameseExtract, điểm LB tốt nhất đến từ các bản dùng mô hình pretrained, viết trước khi đội chuyển sang luật không dùng pretrained. Code các bản đó không còn giữ, README của từng bài mô tả lại cách làm và số liệu.

Mỗi thư mục có README riêng mô tả bài toán, cách giải chi tiết, số liệu, các hướng đã thử và bài học.

### Lưu ý

- Repo **không chứa dữ liệu** và trọng số. Đặt dữ liệu của từng bài vào đúng thư mục bài trước khi chạy (xem mục "Chạy" trong README của bài). Chỉ kèm vài file nộp đã biết đúng điểm LB, trong thư mục `submissions/` của bài.
- Một số bài cần `scorer.py` đi kèm gói dữ liệu của bài.
- Điểm "val" là điểm tự đánh giá trên tập validation; điểm "LB"/"public" là điểm trên bảng xếp hạng public của hệ thống ôn luyện.
