# Dạng 11 - Hỏi đáp trích xuất (đọc hiểu, trích span)

## 1. Nhận dạng đề

- Đầu vào: đoạn văn + câu hỏi. Đầu ra: cụm từ **chép nguyên văn** từ đoạn văn, hoặc rỗng nếu không có câu trả lời.
- Đã gặp: VietnameseExtract / VIQA (Jaccard âm tiết, có thêm SQuAD tiếng Anh), ViQuAD_ExtractiveQA (tự host, 30% không có đáp án).
- Luật: không pretrained NLP. Dữ liệu tiếng Việt thường ít, nên mạng đọc hiểu nơ-ron train từ đầu yếu.

## 2. Metric

F1 theo từ / Jaccard / Exact Match. Câu không có đáp án: để trống mới được điểm. Xem [01_metric.md](../01_metric.md).

## 3. Lộ trình

| Bước | Việc | Kỳ vọng |
|---|---|---|
| 0 | Baseline BTC, nộp; tính điểm "để trống hết" trên dev | ViQuAD: để trống hết khoảng 0,3 |
| 1 | **Chọn câu**: tách câu, điểm câu = tổng IDF các từ câu hỏi xuất hiện trong câu; lấy 1-2 câu tốt nhất | |
| 2 | **Sinh span ứng viên** 1-10 âm tiết trong các câu đó (không bắt đầu / kết thúc bằng dấu câu, không toàn từ của câu hỏi) | 92% đáp án dài ≤ 10 âm tiết |
| 3 | **Đặc trưng span + xếp hạng** bằng XGBoost / LightGBM (`XGBRanker` nhóm theo câu hỏi, nhãn = F1 với đáp án) | phần lớn điểm đến từ đây |
| 4 | **Ngưỡng để trống**: nếu điểm span tốt nhất < t thì trả rỗng; chọn t trên dev | rất quan trọng khi có câu không đáp án |
| 5 | (Nhiều dữ liệu) BiLSTM / Transformer nhỏ dự đoán start/end + lớp "không đáp án" | |

## 4. Đặc trưng span (đã dùng ở VIQA, 37 đặc trưng)

- Câu chứa span: hạng câu, điểm IDF của câu, tỉ lệ từ câu hỏi trong câu.
- Span: độ dài, tỉ lệ từ của câu hỏi trong span (càng thấp càng tốt), số chữ số, số từ viết hoa, có chứa "năm", "%", đơn vị.
- Vị trí: khoảng cách tới từ câu hỏi gần nhất bên trái và bên phải; số từ câu hỏi và tổng IDF trong cửa sổ 6 token quanh span; từ ngay trước / sau span có thuộc câu hỏi không.
- Loại câu hỏi (cờ): "bao nhiêu", "khi nào", "năm nào", "ai", "ở đâu", "gì", "tại sao", "như thế nào", "nào"... Kết hợp với đặc trưng span (câu hỏi "bao nhiêu" thì span có chữ số).

```python
def sinh_span(cau_tokens, hoi_set, Lmax=10):
    out = []
    for i in range(len(cau_tokens)):
        for j in range(i + 1, min(i + Lmax, len(cau_tokens)) + 1):
            sp = cau_tokens[i:j]
            if not re.match(r"\w", sp[0]) or not re.match(r"\w", sp[-1]): continue
            if all(t.lower() in hoi_set for t in sp): continue
            out.append((i, j))
    return out

import xgboost as xgb
rk = xgb.XGBRanker(objective="rank:pairwise", n_estimators=400, learning_rate=0.05, max_depth=6)
rk.fit(X, nhan_0_4, group=so_span_moi_cau_hoi)      # nhan: F1 với đáp án quy về 0-4; group: số span của từng câu hỏi theo thứ tự
```

Lấy span: trả về **chuỗi gốc** trong đoạn văn từ vị trí ký tự đầu đến cuối (giữ đúng dấu, chữ hoa), không ghép lại token.

## 5. Nút vặn

1. Ngưỡng để trống (nếu có câu không đáp án).
2. Đặc trưng loại câu hỏi x loại span.
3. Thêm dữ liệu được cấp ở ngôn ngữ khác qua đặc trưng không phụ thuộc ngôn ngữ (VIQA: 3.000 mẫu SQuAD tiếng Anh).
4. Cắt bớt dấu câu / từ nối ở đầu cuối span ("là", "của", "và").
5. Chọn 2-3 câu thay vì 1 (đáp án có khi ở câu kế bên).

## 6. Bẫy

- Đoạn văn trùng nhau giữa train và test: chia val **theo bài / đoạn văn**, không theo câu hỏi.
- NFC; khoảng trắng đặc biệt; dấu ngoặc kép kiểu Unicode.
- Câu hỏi không đáp án được viết để giống câu hỏi thật: mô hình hay "tự tin sai"; dùng chênh lệch điểm span tốt nhất và nhì làm đặc trưng cho quyết định để trống.

## 7. Bài luyện

- Code tay: [d03_hoi_dap](../../luyen_tap/CV/d03_hoi_dap/) (chọn câu IDF, sinh span, LightGBM / XGBoost xếp hạng, ngưỡng để trống).
- Thi thử: [ViQuAD_ExtractiveQA](../../selfhost_kaggle/ViQuAD_ExtractiveQA/) (baseline 0,276 / 0,337).
- Lời giải tham khảo: [challenges/VietnameseExtract](../../challenges/VietnameseExtract/) (sinh span + XGBRanker, đặc trưng chi tiết).

## 8. Đọc thêm

- Rajpurkar và cộng sự, *Know What You Don't Know: Unanswerable Questions for SQuAD* ([arXiv:1806.03822](https://arxiv.org/abs/1806.03822)).
- Seo và cộng sự, *Bidirectional Attention Flow* (BiDAF, [arXiv:1611.01603](https://arxiv.org/abs/1611.01603)): mô hình đọc hiểu không cần pretrained.
- Nguyen và cộng sự, *UIT-ViQuAD* ([arXiv:2009.14725](https://arxiv.org/abs/2009.14725)).
