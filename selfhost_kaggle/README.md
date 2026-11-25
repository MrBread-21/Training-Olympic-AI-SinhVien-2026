# Self-host Kaggle: các cuộc thi đội tự dựng

Đề thi thử do đội tự dựng và host dưới dạng cuộc thi **Kaggle Community** riêng tư, mô phỏng thể thức OlpAI: 6 tiếng, có public và private test, giới hạn số lần nộp, chỉ backbone ImageNet, không mô hình ngôn ngữ huấn luyện sẵn.

Thư mục này chứa **đề** và [`tools/`](tools/) để dựng lại đề. Lời giải nằm ở [`challenges/`](../challenges/) cùng tên bài.

## Buổi thi thử 26/11/2026 (tác giả đề: MrBread)

| Bài | Nội dung | Metric | Baseline (public / private) |
|---|---|---|---|
| [Nordland_PlaceRecognition](Nordland_PlaceRecognition/) | Tìm ảnh mùa hè cùng vị trí với ảnh mùa đông trên tuyến đường sắt Nordland (làm lại tác vụ 2 vòng loại OlpAI'26) | Recall@1 | 0.069 / 0.085 |
| [FGVC_Aircraft_Noisy](FGVC_Aircraft_Noisy/) | Phân loại 70 họ máy bay, 15% nhãn train sai, 60% ảnh test bị làm hỏng | Macro-F1 | 0.338 / 0.329 |
| [Kvasir_PolypSegmentation](Kvasir_PolypSegmentation/) | Phân đoạn polyp trong ảnh nội soi, nộp RLE | Dice | 0.508 / 0.505 |
| [MVTec_AnomalyDetection](MVTec_AnomalyDetection/) | Phát hiện sản phẩm lỗi khi train chỉ có ảnh tốt, 5 loại sản phẩm | AUROC | 0.642 / 0.635 |
| [UCF101_ActionRecognition](UCF101_ActionRecognition/) | Nhận dạng 10 hành động trong video, test là video gốc chưa gặp | Macro-F1 | 0.864 / 0.881 |
| [VOC_MultiLabel](VOC_MultiLabel/) | Phân loại đa nhãn 20 lớp, chọn ngưỡng từng lớp | Macro-F1 | 0.714 / 0.713 |
| [VSFC_Sentiment](VSFC_Sentiment/) | Cảm xúc phản hồi sinh viên, lớp trung tính rất hiếm | Macro-F1 | 0.664 / 0.649 |
| [PhoNER_COVID19](PhoNER_COVID19/) | Nhận dạng 10 loại thực thể trong tin COVID-19 | F1 thực thể | 0.428 / 0.421 |
| [Vietnamese_Diacritics](Vietnamese_Diacritics/) | Thêm dấu cho câu tiếng Việt không dấu | Độ chính xác từ | 0.754 / 0.751 |
| [IWSLT_EnVi_Translation](IWSLT_EnVi_Translation/) | Dịch Anh sang Việt (TED), không pretrained | BLEU-4 | 2.32 / 1.84 |
| [ViQuAD_PassageRetrieval](ViQuAD_PassageRetrieval/) | Tìm đoạn Wikipedia chứa câu trả lời | MRR@10 | 0.544 / 0.558 |

Baseline là `baseline.py` trong từng bài, giống notebook mẫu BTC phát lúc thi; điểm của nó dùng làm Min_Score khi quy đổi.

## Mỗi bài gồm

- `README.md`: đề (Overview, Evaluation, Data).
- `KAGGLE.md`: nội dung tách sẵn theo từng ô khi tạo cuộc thi trên Kaggle (Title, URL, Subtitle, Overview, Description, Evaluation, Timeline, Rules, Citation, Dataset Description, Data license).
- `baseline.py`: baseline chạy được ngay, ghi `submission.csv`.

Dữ liệu và đáp án (`solution.csv` có cột `Usage` chia Public/Private) không nằm trong repo vì dữ liệu thuộc giấy phép của bên thứ ba (đa số chỉ dùng phi thương mại). Dựng lại bằng [`tools/`](tools/): tải dữ liệu gốc, dựng đề, kiểm tra metric, chạy baseline, nén gói.

## Luật chung (giống vòng khu vực OlpAI'26)

- 6 tiếng; tối đa 20 lần nộp; chọn 2 bài nộp cuối để chấm private.
- CV: chỉ weights ImageNet trong danh sách của BTC:
  - torchvision (`weights="DEFAULT"`): resnet18, resnet34, resnet50, convnext_tiny, efficientnet_b0, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, regnet_y_400mf, densenet121, shufflenet_v2, squeezenet.
  - timm / Hugging Face: resnet18.a1_in1k, resnet50.a1_in1k, efficientnet_b0, mobilenetv3, microsoft/resnet.
  - Không CLIP, SigLIP, DINOv2, VideoMAE, SAM.
- NLP: không có pretrained nào; `transformers` chỉ dùng kiến trúc.
- Không dùng dữ liệu ngoài, không gán nhãn tay cho test.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Điểm quy đổi: `Norm = (điểm - Min_Score) / (Max_Score - Min_Score) x 100`, Min_Score là điểm baseline, Max_Score là điểm private cao nhất của buổi thi.

## Host một bài trên Kaggle

1. kaggle.com/competitions, **Host a Competition**, loại **Community**, chế độ **Private**; điền Title, URL, Subtitle theo `KAGGLE.md`.
2. **Metric**: tạo Metric Notebook mới, dán toàn bộ hàm chấm của bài (lấy từ [`metrics.py`](metrics.py): phần chung đầu file + hàm của bài + hàm `score`), bấm Save & Validate, rồi chọn metric đó trong Settings > Evaluation Metric.
3. **Data**: tải gói dữ liệu của bài; Kaggle đặt nó ở `/kaggle/input/competitions/<url>/`.
4. **Solution**: tải `solution.csv` (cột id `id`, cột chia `Usage`), kiểm tra bằng Sandbox Submission với file của `baseline.py`.
5. Dán các ô chữ trong `KAGGLE.md` vào Overview, Rules, Data; chọn Data license.
6. **Key Dates** tính theo UTC (giờ Việt Nam trừ 7); giới hạn nộp mỗi ngày reset lúc 00:00 UTC.
7. Gửi Invitation Link cho đội. Notebook gắn với cuộc thi riêng tư chỉ chia sẻ được cho collaborator hoặc Group, không để Public được.

| Metric | Hàm trong `metrics.py` | Dùng cho |
|---|---|---|
| Recall@1, dung sai 2 khung | `recall_at_1` (+ `_recall`) | Nordland_PlaceRecognition |
| Macro-F1 | `macro_f1` | FGVC_Aircraft_Noisy, UCF101_ActionRecognition, VSFC_Sentiment |
| Dice trung bình, mask RLE | `mean_dice` (+ `rle_decode`) | Kvasir_PolypSegmentation |
| AUROC | `auroc` | MVTec_AnomalyDetection |
| Macro-F1 đa nhãn | `multilabel_macro_f1` | VOC_MultiLabel |
| F1 mức thực thể | `entity_f1` | PhoNER_COVID19 |
| Độ chính xác từ | `word_accuracy` | Vietnamese_Diacritics |
| BLEU-4 corpus | `bleu4` (+ `_tach`) | IWSLT_EnVi_Translation |
| MRR@10 | `mrr_at_10` | ViQuAD_PassageRetrieval |
