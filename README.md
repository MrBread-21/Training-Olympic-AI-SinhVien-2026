# Bộ bài luyện NLP (nhánh feature/hue)

Nhánh luyện tập NLP cho kỳ thi Olympic Trí tuệ nhân tạo sinh viên Việt Nam 2026 (OlpAI'26): 16 bài chia thành 8 nhóm dạng bài, mỗi nhóm 2 bài, lấy từ các cuộc thi Kaggle, DrivenData và bộ dữ liệu nghiên cứu (có nhiều bài tiếng Việt).

Hiện nhánh mới có danh sách đề; mô tả, metric và link dữ liệu từng bài nằm trong [`NLP/README.md`](NLP/README.md).

## Danh sách bài

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

## Cấu trúc

```
NLP/
└── README.md     danh sách 16 bài: mô tả, metric, link đề và dữ liệu, trạng thái cuộc thi
```

Các nhánh khác của repo được mô tả trong README của nhánh `main`.
