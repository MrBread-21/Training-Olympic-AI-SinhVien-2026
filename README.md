# Training Olympic AI Sinh viên 2026

Repo luyện tập của đội cho kỳ thi Olympic Trí tuệ nhân tạo sinh viên Việt Nam 2026 (OlpAI'26). Nhánh `main` tổng hợp bài của các thành viên; mỗi thành viên vẫn làm trên nhánh riêng rồi gộp vào `main`.

## Tác giả từng phần

| Thư mục | Nội dung | Tác giả | Nhánh gốc |
|---|---|---|---|
| [`luyen_tay/`](luyen_tay/) | Bộ bài luyện code tay theo ngày (20/11 đến 30/11) | MrBread ([@MrBread-21](https://github.com/MrBread-21)) | [`MrBread`](../../tree/MrBread) |
| [`EuroSAT/`](EuroSAT/) | Lời giải bài ôn luyện Phenikaa: phân loại ảnh vệ tinh | MrBread ([@MrBread-21](https://github.com/MrBread-21)) | [`MrBread`](../../tree/MrBread) |
| [`ImgCap/`](ImgCap/) | Lời giải bài ôn luyện Phenikaa: mô tả ảnh | MrBread ([@MrBread-21](https://github.com/MrBread-21)) | [`MrBread`](../../tree/MrBread) |
| [`VietnameseExtract/`](VietnameseExtract/) | Lời giải bài ôn luyện Phenikaa: hỏi đáp trích xuất tiếng Việt | MrBread ([@MrBread-21](https://github.com/MrBread-21)) | [`MrBread`](../../tree/MrBread) |
| [`CCTV_Street_Scence/`](CCTV_Street_Scence/) | Lời giải bài ôn luyện Phenikaa: bất thường trong video giám sát | MrBread ([@MrBread-21](https://github.com/MrBread-21)) | [`MrBread`](../../tree/MrBread) |
| [`MSAD/`](MSAD/) | Lời giải bài ôn luyện Phenikaa: phân loại và định vị sự kiện bất thường | MrBread ([@MrBread-21](https://github.com/MrBread-21)) | [`MrBread`](../../tree/MrBread) |
| [`NLP/`](NLP/) | Bộ 16 bài luyện NLP (8 nhóm dạng bài) | Hue ([@nguyenthithanhhue-13](https://github.com/nguyenthithanhhue-13)) | [`feature/hue`](../../tree/feature/hue) |

---

## Phần của MrBread

### Bộ bài luyện code tay (`luyen_tay/`)

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

### Lời giải các bài ôn luyện PHENIKAA AI Hackathon

Lời giải cho các bài trên [hệ thống ôn luyện PHENIKAA AI Hackathon](https://aihackathon.phenikaa-uni.edu.vn/). Phần lớn viết theo luật của kỳ thi: **không dùng trọng số pretrained** (chỉ dùng kiến trúc, trọng số tự train trên dữ liệu được cấp), hoặc chỉ dùng backbone ImageNet trong danh sách được cấp và để đóng băng. Hầu hết chạy được trên GPU 4 GB (GTX 1650, fp32) trong khoảng 20 phút.

| Bài | Nhiệm vụ | Hướng giải chính | LB public tốt nhất |
|---|---|---|---|
| [EuroSAT](EuroSAT/) | Phân loại ảnh vệ tinh 10 lớp khi ít nhãn, ảnh test bị làm hỏng | Augmentation mô phỏng biến đổi của test, chọn model trên val bị làm hỏng, TTA 8 phép xoay/lật; bản trong repo dùng ResNet-18 train từ đầu + pseudo-label | **0.8074** (bản fine-tune ResNet-50 pretrained); bản trong repo: val hard 0.665 |
| [ImgCap](ImgCap/) | Mô tả ảnh (COCO, test nocaps) | Bản trong repo: ResNet-18 train từ đầu dự đoán 400 từ khái niệm, truy hồi 20 ảnh gần nhất, chọn câu đồng thuận (MBR theo CIDEr) | **1.1927** (bản BLIP pretrained); bản trong repo: val CIDEr-D 0.2725 |
| [VietnameseExtract](VietnameseExtract/) | Hỏi đáp trích xuất tiếng Việt | Bản trong repo: chọn câu theo IDF, sinh span ứng viên, xếp hạng bằng XGBRanker, thêm SQuAD tiếng Anh | **0.6517** (bản XLM-R pretrained); bản trong repo: val Jaccard 0.227 |
| [CCTV_Street_Scence](CCTV_Street_Scence/) | Điểm bất thường từng khung, camera giám sát cố định | Điểm mức video (đường vắng) + kNN của blob vật thể so với video bình thường, chạy CPU, không pretrained | **0.6761**; `solve.py` mặc định: 0.6727 |
| [MSAD](MSAD/) | Phân loại và định vị sự kiện bất thường trong video | Đặc trưng chuyển động/màu thủ công; mô hình bảng + CNN train từ đầu, hoặc backbone ImageNet đóng băng + LR + TCN | **0.4149** (`main_scratch.py`, không pretrained) |

Với EuroSAT, ImgCap và VietnameseExtract, điểm LB tốt nhất đến từ các bản dùng mô hình pretrained, viết trước khi chuyển sang luật không dùng pretrained. Code các bản đó không còn giữ, README của từng bài mô tả lại cách làm và số liệu.

Mỗi thư mục có README riêng mô tả bài toán, cách giải chi tiết, số liệu, các hướng đã thử và bài học. Repo không chứa dữ liệu và trọng số; chỉ kèm vài file nộp đã biết đúng điểm LB trong thư mục `submissions/` của bài. Một số bài cần `scorer.py` đi kèm gói dữ liệu của bài. Điểm "val" là điểm tự đánh giá trên tập validation; điểm "LB"/"public" là điểm trên bảng xếp hạng public của hệ thống ôn luyện.

---

## Phần của Hue

### Bộ bài luyện NLP (`NLP/`)

16 bài chia thành 8 nhóm dạng bài, mỗi nhóm 2 bài, lấy từ các cuộc thi Kaggle, DrivenData và bộ dữ liệu nghiên cứu (có nhiều bài tiếng Việt). Hiện mới có danh sách đề; mô tả, metric và link dữ liệu từng bài nằm trong [`NLP/README.md`](NLP/README.md).

| Nhóm | Bài 1 | Bài 2 |
|---|---|---|
| 01. Phân loại văn bản | Disaster Tweets (F1) | Jigsaw Toxic Comments, đa nhãn (ROC-AUC trung bình) |
| 02. Cảm xúc theo khía cạnh (ABSA) | UIT-ViSFD, tiếng Việt (macro-F1) | UIT-ABSA (aspect F1, sentiment F1) |
| 03. Chuẩn hoá văn bản | Khôi phục dấu tiếng Việt (độ chính xác ký tự / từ) | Sửa lỗi chính tả tiếng Việt (exact match, độ chính xác từ) |
| 04. Trích xuất, nhận diện tác giả | Tweet Sentiment Extraction (Jaccard) | Name That LLM (balanced accuracy) |
| 05. Sinh văn bản, suy luận | Tóm tắt bài báo khoa học (ROUGE-2 F1) | Contradictory, My Dear Watson, NLI đa ngôn ngữ (accuracy) |
| 06. Biểu diễn, truy hồi | U.S. Patent Phrase Matching (Pearson) | Quora Question Pairs (log loss) |
| 07. Đánh giá LLM, kiểm chứng thông tin | LLM Classification Finetuning (log loss) | ViFactCheck, tiếng Việt (F1) |
| 08. Bài kiểu IOAI | IOAI 2025: phát hiện ảo giác (hallucination) | IOAI 2026: bài Potato |

---

## Quy ước

- Mỗi thành viên làm trên nhánh của mình rồi gộp vào `main` bằng merge (giữ nguyên commit và tên tác giả), không đẩy thẳng lên `main`.
- Không đưa dữ liệu của ban tổ chức, trọng số mô hình và file cache lên repo; README mỗi bài ghi rõ cần đặt dữ liệu ở đâu.
- Mỗi bài có README nêu: bài toán, cách giải, số liệu (val và LB), những gì đã thử, bài học.
