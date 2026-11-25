# Training Olympic AI Sinh viên 2026

Repo luyện tập của đội cho kỳ thi Olympic Trí tuệ nhân tạo sinh viên Việt Nam 2026 (OlpAI'26): bài luyện theo chủ đề, đề thi thử đội tự dựng và host trên Kaggle, và lời giải các bài thi thử. Mọi lời giải theo luật của kỳ thi: chỉ backbone ImageNet trong danh sách của BTC, không mô hình ngôn ngữ huấn luyện sẵn, chạy được trên GPU nhỏ trong thời gian giới hạn.

## Cấu trúc

```
luyen_tap/                         bài luyện theo chủ đề
├── CV/                            bộ bài luyện code tay CV theo ngày (20/11 đến 30/11)       MrBread
└── NLP/                           bộ 16 bài luyện NLP                                         Hue
selfhost_kaggle/                   ĐỀ các cuộc thi đội tự host trên Kaggle                      MrBread
├── tools/                         script tải dữ liệu, dựng đề, kiểm tra metric, nén gói
├── metrics.py                     hàm chấm theo giao diện Kaggle Metric Notebook
└── <19 bài>/                      đề, nội dung từng ô Kaggle, baseline
challenges/                        LỜI GIẢI                                                     MrBread
├── EuroSAT/  ImgCap/  VietnameseExtract/  CCTV_Street_Scence/  MSAD/    bài thi thử PHENIKAA
└── Nordland_PlaceRecognition/     bài tự host trong selfhost_kaggle/
```

Đề nằm ở `selfhost_kaggle/<tên bài>/`, lời giải của bài đó nằm ở `challenges/<cùng tên bài>/`.

## Tác giả từng phần

| Thư mục | Nội dung | Tác giả |
|---|---|---|
| [`luyen_tap/CV/`](luyen_tap/CV/) | Bộ bài luyện code tay CV theo ngày | MrBread ([@MrBread-21](https://github.com/MrBread-21)) |
| [`luyen_tap/NLP/`](luyen_tap/NLP/) | Bộ 16 bài luyện NLP (8 nhóm dạng bài) | Hue ([@nguyenthithanhhue-13](https://github.com/nguyenthithanhhue-13)) |
| [`selfhost_kaggle/`](selfhost_kaggle/) | Đề 19 cuộc thi Kaggle tự host và script dựng đề | MrBread ([@MrBread-21](https://github.com/MrBread-21)) |
| [`challenges/`](challenges/) | Lời giải 5 bài PHENIKAA và bài Nordland | MrBread ([@MrBread-21](https://github.com/MrBread-21)) |

Mỗi thành viên làm trên nhánh riêng ([`MrBread`](../../tree/MrBread), [`feature/hue`](../../tree/feature/hue)) rồi gộp vào `main`.

---

## Luyện tập (`luyen_tap/`)

### CV: bộ bài luyện code tay (`luyen_tap/CV/`) - MrBread

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

Trọng tâm là CV; ba ngày cuối (NLP, seq2seq, dữ liệu bảng) là phần bổ trợ để không bị bất ngờ nếu đề ra ngoài CV. Ngưỡng được đặt để tách cách làm đúng khỏi cách làm sơ sài, ví dụ: giữ nguyên khung bài 21 cho bài 22 chỉ được 0.82, quên chuẩn hoá NFC ở bài 28 chỉ được 0.80, chỉ dùng FGSM ở bài 25 chỉ được 0.42.

```
cd luyen_tap/CV/d21_phan_loai_anh
python tao_du_lieu.py              # tạo data/ và .dap_an/ (không đưa lên repo)
python check_21.py bai_lam.py      # chạy bài làm rồi chấm
```

Xem thêm [`luyen_tap/CV/README.md`](luyen_tap/CV/README.md). Một số đề nhắc tới `ON_TAP.md`: đây là ghi chú ôn tập cá nhân, không có trong repo.

### NLP: bộ 16 bài luyện (`luyen_tap/NLP/`) - Hue

16 bài chia thành 8 nhóm dạng bài, mỗi nhóm 2 bài, lấy từ các cuộc thi Kaggle, DrivenData và bộ dữ liệu nghiên cứu (có nhiều bài tiếng Việt). Hiện mới có danh sách đề; mô tả, metric và link dữ liệu từng bài nằm trong [`luyen_tap/NLP/README.md`](luyen_tap/NLP/README.md).

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

## Self-host Kaggle (`selfhost_kaggle/`) - MrBread

Đề thi thử đội tự dựng và host dưới dạng cuộc thi **Kaggle Community** riêng tư, mô phỏng thể thức OlpAI: 6 tiếng, public và private test, tối đa 20 lần nộp, chọn 2 bài cuối, `main.py` chạy lại dưới 20 phút. Buổi đầu tiên: **26/11/2026**, 11 bài; thêm 2 bài DimABSA dựng lại đề NLP OlpAI'26 và 6 bài lấp các dạng còn thiếu (hỏi đáp trích xuất, caption, tấn công đối kháng, âm thanh, dữ liệu bảng, đường đi robot).

| Bài | Nội dung | Metric | Baseline (public) |
|---|---|---|---|
| [Nordland_PlaceRecognition](selfhost_kaggle/Nordland_PlaceRecognition/) | Với mỗi ảnh mùa hè (bị làm hỏng), tìm ảnh mùa đông cùng vị trí; dựng theo đúng đề tác vụ 2 vòng sơ loại OlpAI'26 | Recall@1 (0-100, ±10 khung) | 4.22 |
| [FGVC_Aircraft_Noisy](selfhost_kaggle/FGVC_Aircraft_Noisy/) | 70 họ máy bay, 15% nhãn train sai, 60% ảnh test bị làm hỏng | Macro-F1 | 0.338 |
| [Kvasir_PolypSegmentation](selfhost_kaggle/Kvasir_PolypSegmentation/) | Phân đoạn polyp nội soi, nộp RLE | Dice | 0.508 |
| [MVTec_AnomalyDetection](selfhost_kaggle/MVTec_AnomalyDetection/) | Phát hiện sản phẩm lỗi, train chỉ có ảnh tốt | AUROC | 0.642 |
| [UCF101_ActionRecognition](selfhost_kaggle/UCF101_ActionRecognition/) | 10 hành động trong video, test là video gốc chưa gặp | Macro-F1 | 0.864 |
| [VOC_MultiLabel](selfhost_kaggle/VOC_MultiLabel/) | Phân loại đa nhãn 20 lớp | Macro-F1 | 0.714 |
| [VSFC_Sentiment](selfhost_kaggle/VSFC_Sentiment/) | Cảm xúc phản hồi sinh viên | Macro-F1 | 0.664 |
| [PhoNER_COVID19](selfhost_kaggle/PhoNER_COVID19/) | Nhận dạng thực thể tin COVID-19 | F1 thực thể | 0.428 |
| [Vietnamese_Diacritics](selfhost_kaggle/Vietnamese_Diacritics/) | Thêm dấu tiếng Việt | Độ chính xác từ | 0.754 |
| [IWSLT_EnVi_Translation](selfhost_kaggle/IWSLT_EnVi_Translation/) | Dịch Anh sang Việt, không pretrained | BLEU-4 | 2.32 |
| [ViQuAD_PassageRetrieval](selfhost_kaggle/ViQuAD_PassageRetrieval/) | Tìm đoạn Wikipedia chứa câu trả lời | MRR@10 | 0.544 |
| [DimABSA_VA](selfhost_kaggle/DimABSA_VA/) | Đề NLP OlpAI'26, Subtask 1: dự đoán (valence, arousal) cho từng khía cạnh cho trước | RMSE_VA (thấp hơn là tốt) | 1.901 |
| [DimABSA_Quad](selfhost_kaggle/DimABSA_Quad/) | Đề NLP OlpAI'26, Subtask 2: trích bộ tứ (aspect, category, opinion, VA) | cF1 | 0.305 |
| [ViQuAD_ExtractiveQA](selfhost_kaggle/ViQuAD_ExtractiveQA/) | Hỏi đáp trích xuất tiếng Việt, có câu hỏi không có đáp án | F1 theo từ | 0.276 |
| [KTVIC_Captioning](selfhost_kaggle/KTVIC_Captioning/) | Sinh caption tiếng Việt cho ảnh đời sống | CIDEr-D | 0.196 |
| [CIFAR_Adversarial](selfhost_kaggle/CIFAR_Adversarial/) | Tấn công đối kháng: lừa mô hình công bố và mô hình giấu kín | Tỉ lệ lừa được | 0.809 |
| [SpeechCommands_KWS](selfhost_kaggle/SpeechCommands_KWS/) | Nhận dạng lệnh nói, người nói mới, có tạp âm | Macro-F1 | 0.377 |
| [HomeCredit_Tabular](selfhost_kaggle/HomeCredit_Tabular/) | Dự đoán vỡ nợ, 3 bảng, chia theo thời gian | AUROC | 0.761 |
| [Robot_PathPlanning](selfhost_kaggle/Robot_PathPlanning/) | Robot ghé mọi điểm mốc với chi phí địa hình nhỏ nhất | Tối ưu / của bạn | 0.777 |

Mỗi bài có `README.md` (đề), `KAGGLE.md` (nội dung tách sẵn từng ô khi tạo cuộc thi) và `baseline.py`. Cách host trên Kaggle: [`selfhost_kaggle/README.md`](selfhost_kaggle/README.md). Cách dựng lại dữ liệu và đáp án: [`selfhost_kaggle/tools/README.md`](selfhost_kaggle/tools/README.md).

---

## Challenges (`challenges/`) - MrBread

### Bài thi thử PHENIKAA AI Hackathon

Lời giải các bài trên [hệ thống ôn luyện PHENIKAA AI Hackathon](https://aihackathon.phenikaa-uni.edu.vn/). Phần lớn viết theo luật của kỳ thi: **không dùng trọng số pretrained** (chỉ dùng kiến trúc, trọng số tự train trên dữ liệu được cấp), hoặc chỉ dùng backbone ImageNet trong danh sách được cấp và để đóng băng. Hầu hết chạy được trên GPU 4 GB (GTX 1650, fp32) trong khoảng 20 phút.

| Bài | Nhiệm vụ | Hướng giải chính | LB public tốt nhất |
|---|---|---|---|
| [EuroSAT](challenges/EuroSAT/) | Phân loại ảnh vệ tinh 10 lớp khi ít nhãn, ảnh test bị làm hỏng | Augmentation mô phỏng biến đổi của test, chọn model trên val bị làm hỏng, TTA 8 phép xoay/lật; bản trong repo dùng ResNet-18 train từ đầu + pseudo-label | **0.8074** (bản fine-tune ResNet-50 pretrained); bản trong repo: val hard 0.665 |
| [ImgCap](challenges/ImgCap/) | Mô tả ảnh (COCO, test nocaps) | Bản trong repo: ResNet-18 train từ đầu dự đoán 400 từ khái niệm, truy hồi 20 ảnh gần nhất, chọn câu đồng thuận (MBR theo CIDEr) | **1.1927** (bản BLIP pretrained); bản trong repo: val CIDEr-D 0.2725 |
| [VietnameseExtract](challenges/VietnameseExtract/) | Hỏi đáp trích xuất tiếng Việt | Bản trong repo: chọn câu theo IDF, sinh span ứng viên, xếp hạng bằng XGBRanker, thêm SQuAD tiếng Anh | **0.6517** (bản XLM-R pretrained); bản trong repo: val Jaccard 0.227 |
| [CCTV_Street_Scence](challenges/CCTV_Street_Scence/) | Điểm bất thường từng khung, camera giám sát cố định | Điểm mức video (đường vắng) + kNN của blob vật thể so với video bình thường, chạy CPU, không pretrained | **0.6761**; `solve.py` mặc định: 0.6727 |
| [MSAD](challenges/MSAD/) | Phân loại và định vị sự kiện bất thường trong video | Đặc trưng chuyển động/màu thủ công; mô hình bảng + CNN train từ đầu, hoặc backbone ImageNet đóng băng + LR + TCN | **0.4149** (`main_scratch.py`, không pretrained) |

Với EuroSAT, ImgCap và VietnameseExtract, điểm LB tốt nhất đến từ các bản dùng mô hình pretrained, viết trước khi chuyển sang luật không dùng pretrained. Code các bản đó không còn giữ, README của từng bài mô tả lại cách làm và số liệu. Một số bài cần `scorer.py` đi kèm gói dữ liệu của bài; vài file nộp đã biết đúng điểm LB nằm trong thư mục `submissions/` của bài.

### Bài tự host

| Bài | Hướng giải chính | Val | Test (public / private) |
|---|---|---|---|
| [Nordland_PlaceRecognition](challenges/Nordland_PlaceRecognition/) | ResNet-18 ImageNet tới layer3, descriptor lưới không gian 4x7, InfoNCE đông-hè với negative là các khung liền kề, đóng băng BatchNorm; train 25 phút trên GTX 1650 | R@1 0.929 | **0.814 / 0.835** (baseline 0.069) |

Lời giải Nordland được viết cho bản đề giả định trước đây (ảnh đông tìm ảnh hè, dung sai ±2 khung, thang 0-1); đề đã được dựng lại theo văn bản chính thức nên các con số trên không so được với đề hiện tại.

Mỗi thư mục có README riêng: bài toán, cách giải, số liệu, các hướng đã thử và bài học. Điểm "val" là điểm tự đánh giá trên tập validation tự chia; điểm "LB"/"public" là điểm trên bảng xếp hạng.

---

## Quy ước

- Bài luyện theo chủ đề đặt trong `luyen_tap/<chủ đề>/`; đề cuộc thi tự host đặt trong `selfhost_kaggle/<tên bài>/`; lời giải đặt trong `challenges/<tên bài>/`, cùng tên với đề nếu là bài tự host.
- Mỗi thành viên làm trên nhánh của mình rồi gộp vào `main` bằng merge (giữ nguyên commit và tên tác giả).
- Không đưa dữ liệu của ban tổ chức hay của bên thứ ba, đáp án, trọng số mô hình và file cache lên repo; README mỗi bài ghi rõ cần đặt dữ liệu ở đâu hoặc cách dựng lại.
- Mỗi bài có README nêu: bài toán, cách giải, số liệu (val và LB), những gì đã thử, bài học.
