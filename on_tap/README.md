# Ôn luyện OlpAI'26 - giáo trình và lộ trình đến chung kết (7-9/12/2026)

Thư mục này là **điểm bắt đầu** của toàn bộ việc ôn luyện. Nó nối bốn loại tài liệu trong repo:

| Loại | Ở đâu | Dùng để |
|---|---|---|
| **Bài học theo dạng bài** | [on_tap/dang_bai/](dang_bai/) | Hiểu một dạng bài: nhận dạng đề, metric, lộ trình trong 6 tiếng, khung code đã chạy thử, nút vặn, bẫy |
| **Bài luyện code tay** | [luyen_tap/CV/](../luyen_tap/CV/) | Gõ lại khung code không nhìn tài liệu, có dữ liệu giả lập, bộ chấm tự động, lời giải mẫu |
| **Cuộc thi thử** | [selfhost_kaggle/](../selfhost_kaggle/) | Làm bài thật trên Kaggle nội bộ: 6 tiếng, 20 lượt public, private mở bằng mật khẩu ở giờ 6 |
| **Lời giải tham khảo** | [challenges/](../challenges/) | So cách làm sau khi tự làm xong |

Ba tài liệu chung, đọc trước:

- [00_quy_trinh_phong_thi.md](00_quy_trinh_phong_thi.md): luật, chia vai, lịch theo phút, notebook khung, `main.py`, checklist nộp bài.
- [01_metric.md](01_metric.md) + [metric_tu_viet.py](metric_tu_viet.py): công thức và bẫy của mọi metric, code numpy đã đối chiếu với sklearn / nltk.
- [02_huan_luyen.md](02_huan_luyen.md): chẩn đoán, validation, tối ưu, augmentation, ensemble, tốc độ, tái lập.
- [prompt_llm/](prompt_llm/): luyện prompt cho chatbot phòng thi (DeepSeek-R1-Distill-Qwen-32B, 2.000 token mỗi cuộc trò chuyện): khung prompt, phần đầu file chuẩn, 16 prompt mẫu đã chấm bằng bộ chấm bài luyện, công cụ gửi prompt.

## Bản đồ dạng bài

| # | Dạng bài | Bài học | Luyện code tay | Thi thử (baseline public) | Lời giải tham khảo |
|---|---|---|---|---|---|
| 1 | Phân loại ảnh (ảnh hỏng, nhãn nhiễu, ít nhãn) | [01](dang_bai/01_phan_loai_anh.md) | d21, d22, d23 | FGVC_Aircraft_Noisy (0,338) | EuroSAT |
| 2 | Đa nhãn, phân đoạn ảnh | [02](dang_bai/02_da_nhan_va_phan_doan.md) | d27 (B) | VOC_MultiLabel (0,714), Kvasir_PolypSegmentation (0,508) | |
| 3 | Phát hiện bất thường | [03](dang_bai/03_bat_thuong.md) | d27 (A) | MVTec_AnomalyDetection (0,642) | CCTV_Street_Scence |
| 4 | Truy xuất ảnh, nhận dạng địa điểm | [04](dang_bai/04_truy_xuat_anh.md) | d24 | **Nordland_PlaceRecognition** (đề sơ loại OlpAI'26; 4,22) | Nordland_PlaceRecognition |
| 5 | Video: phân loại, định vị sự kiện | [05](dang_bai/05_video.md) | d26 | UCF101_ActionRecognition (0,864) | MSAD |
| 6 | Tấn công đối kháng | [06](dang_bai/06_doi_khang.md) | d25 | CIFAR_Adversarial (0,809) | CIFAR_Adversarial (0,962) |
| 7 | Sinh caption | [07](dang_bai/07_caption.md) | **d02** | KTVIC_Captioning (0,196) | ImgCap, KTVIC_Captioning (1,029) |
| 8 | Âm thanh | [08](dang_bai/08_am_thanh.md) | **d01** | SpeechCommands_KWS (0,377) | SpeechCommands_KWS (0,932) |
| 9 | Phân loại văn bản | [09](dang_bai/09_phan_loai_van_ban.md) | d28 (A, A2) | VSFC_Sentiment (0,664) | |
| 10 | Gán nhãn chuỗi (NER) | [10](dang_bai/10_gan_nhan_chuoi.md) | d04 (B) | PhoNER_COVID19 (0,428) | |
| 11 | Hỏi đáp trích xuất | [11](dang_bai/11_hoi_dap_trich_xuat.md) | **d03** | ViQuAD_ExtractiveQA (0,276) | VietnameseExtract; ViQuAD_ExtractiveQA (thử nghiệm, chưa vượt baseline) |
| 12 | Dịch máy, thêm dấu (seq2seq) | [12](dang_bai/12_seq2seq.md) | d29 | IWSLT_EnVi_Translation (2,32), Vietnamese_Diacritics (0,754) | |
| 13 | Truy xuất văn bản | [13](dang_bai/13_truy_xuat_van_ban.md) | d28 (B) | ViQuAD_PassageRetrieval (0,544) | |
| 14 | Cảm xúc đa chiều theo khía cạnh | [14](dang_bai/14_dimabsa.md) | **d04** | **DimABSA_VA** (1,901, thấp là tốt), **DimABSA_Quad** (0,305) (đề NLP OlpAI'26) | DimABSA_VA (1,863), DimABSA_Quad (0,514) |
| 15 | Dữ liệu bảng | [15](dang_bai/15_du_lieu_bang.md) | d30 | HomeCredit_Tabular (0,761) | HomeCredit_Tabular (0,779) |
| 16 | Tìm đường, học tăng cường | [16](dang_bai/16_do_thi_robot.md) | **d05** | Robot_PathPlanning (0,777) | Robot_PathPlanning (1,000) |

Bài luyện code tay **in đậm** là bài mới cho các ngày 1-5/12 (dạng bài chưa có bài luyện). Điểm trong cột lời giải tham khảo là điểm public của lời giải; **chỉ mở sau khi đã tự làm**.

## Lộ trình 26/11 - 6/12

Mỗi ngày gồm ba phần:

- **Sáng, 1 tiếng**: bài luyện code tay của ngày, bấm giờ, đóng tài liệu (cách làm trong [luyen_tap/CV/README.md](../luyen_tap/CV/README.md)).
- **Chiều, 2-3 tiếng**: đọc bài học, rồi làm cuộc thi thử của dạng đó tới khi vượt baseline rõ. Ghi sổ nộp bài.
- **Tối, 30 phút**: ghi 3 điều học được vào bảng tiến độ cuối file này.

| Ngày | Code tay (sáng) | Bài học | Thi thử (chiều) |
|---|---|---|---|
| T5 26/11 | d26_video | 05, 00 | Buổi thi thử đội: Nordland + bài tự chọn |
| T6 27/11 | d27_patchcore_unet | 03, 02 | MVTec_AnomalyDetection hoặc Kvasir_PolypSegmentation |
| T7 28/11 | d28_nlp | 09, 13, 01 | VSFC_Sentiment, ViQuAD_PassageRetrieval |
| CN 29/11 | d29_seq2seq (sáng sớm) | 12 | **Thi thử 6 tiếng #1 đúng luật**: KTVIC_Captioning + ViQuAD_ExtractiveQA (một CV, một NLP như đề chung kết 2025) |
| T2 30/11 | d30_du_lieu_bang | 15, 02 | Chữa bài thi thử #1 (đọc lời giải, làm lại phần yếu); HomeCredit_Tabular |
| T3 1/12 | **d01_am_thanh** | 08 | SpeechCommands_KWS |
| T4 2/12 | **d02_caption** | 07 | KTVIC_Captioning (làm lại, mục tiêu vượt điểm buổi #1) |
| T5 3/12 | **d03_hoi_dap** | 11, 10 | ViQuAD_ExtractiveQA, PhoNER_COVID19 |
| T6 4/12 | **d04_dimabsa** | 14 | DimABSA_VA, DimABSA_Quad |
| T7 5/12 | **d05_duong_di** | 06, 16 | **Thi thử 6 tiếng #2 đúng luật**: CIFAR_Adversarial + DimABSA_Quad + SpeechCommands_KWS (3 bài, đội 3 người) |
| CN 6/12 | Viết lại nhanh 2 khung yếu nhất | 00, 01, 02 | Nghỉ; kiểm tra giấy tờ, luật chung kết, phân vai; ngủ sớm |

### Thi thử đúng luật

- Một người làm "BTC": mở cuộc thi Kaggle, giữ `host/mat_khau_private_test.txt`, công bố mật khẩu đúng đầu giờ 6.
- Tắt mạng ngoài Kaggle (hoặc tự giác), không LLM (hoặc tối đa 2.000 token), không mở `challenges/` và `loi_giai/`.
- Tối đa 20 lượt public; giờ 6 nộp private; cuối giờ có `main.py` chạy dưới 20 phút.
- Sau buổi: so điểm quy đổi `(điểm - baseline) / (mốc mạnh - baseline)` của từng bài, đọc lời giải tham khảo, viết 5 dòng rút kinh nghiệm.

## Mục tiêu đo được trước chung kết

- [ ] Gõ notebook khung 8 khối ([00](00_quy_trinh_phong_thi.md) mục 4) trong ≤ 15 phút, chạy không lỗi.
- [ ] Đạt ngưỡng mọi bài luyện code tay d21-d30 và d01-d05 trong thời gian mục tiêu.
- [ ] Mỗi cuộc thi thử: vượt baseline trong ≤ 45 phút đầu.
- [ ] Hai buổi thi thử 6 tiếng: có `main.py` tái tạo đúng file nộp tốt nhất, chạy dưới 20 phút.
- [ ] Giải thích được bằng lời: bias / variance, BatchNorm, attention, AdamW và lịch lr, label smoothing, macro / micro F1, AUROC và AP, BLEU và CIDEr, GroupKFold, rò rỉ dữ liệu.

## Bảng tiến độ

| Ngày | Bài | Thời gian | Điểm | Điều học được |
|---|---|---|---|---|
| | | | | |
