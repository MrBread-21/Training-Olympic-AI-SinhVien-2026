# VietnameseExtract - Hỏi đáp trích xuất tiếng Việt

Bài luyện tập VIQA trên [hệ thống ôn luyện PHENIKAA AI Hackathon](https://aihackathon.phenikaa-uni.edu.vn/).

## Bài toán

- Cho đoạn văn Wikipedia tiếng Việt và câu hỏi, trả về đoạn trả lời copy nguyên văn từ đoạn văn.
- Dữ liệu: 87 599 mẫu SQuAD v1.1 tiếng Anh, 1190 mẫu tiếng Việt (XQuAD-vi), 511 mẫu val (MLQA-vi), 2000 mẫu test.
- Điểm: Jaccard mức âm tiết sau khi chữ thường và bỏ dấu câu (phụ: Exact Match).

Lời giải trong `solution.ipynb` **không dùng trọng số pretrained**, theo luật của kỳ thi OlpAI mà đội đang luyện (không có mô hình ngôn ngữ pretrained nào được cấp).

## Ý tưởng

1190 mẫu tiếng Việt quá ít để train một mô hình đọc hiểu nơ-ron từ đầu, nên bài toán được chuyển thành **sinh span ứng viên rồi xếp hạng** bằng cây quyết định:

1. **Chọn câu**: tách câu, tính IDF trên toàn bộ câu của các context, lấy 2 câu có tổng IDF của từ câu hỏi xuất hiện trong câu lớn nhất.
2. **Sinh span**: mọi đoạn dài 1-10 âm tiết trong 2 câu đó, không bắt đầu hoặc kết thúc bằng dấu câu, không gồm toàn từ của câu hỏi. 92.4% đáp án dài không quá 10 âm tiết (trung vị 3).
3. **Đặc trưng cho từng span** (37 đặc trưng): hạng và điểm của câu chứa span, độ dài, tỉ lệ từ câu hỏi trong span, số chữ số, số từ viết hoa, khoảng cách tới từ câu hỏi gần nhất bên trái và bên phải, số từ câu hỏi và tổng IDF trong cửa sổ 6 token quanh span, từ ngay trước và sau có thuộc câu hỏi không, span có chữ "năm" không, và 18 cờ loại câu hỏi ("bao nhiêu", "khi nào", "năm nào", "ai", "ở đâu", "gì", "tại sao"...).
4. **XGBoost**: nhãn của span là Jaccard với đáp án; so sánh hồi quy Jaccard với `XGBRanker` pairwise theo từng câu hỏi (nhãn liên quan 0-4). Chọn span có điểm cao nhất mỗi câu hỏi.
5. **Dùng thêm SQuAD tiếng Anh**: 3000 mẫu, chỉ dùng các đặc trưng cấu trúc không phụ thuộc ngôn ngữ; từ hỏi tiếng Anh được ánh xạ sang loại câu hỏi tiếng Việt (how many thành "bao nhiêu", when thành "khi nào", who thành "ai"...).
6. Bản nộp: train lại trên train_vi + val (+ tiếng Anh nếu biến thể đó thắng trên val).

## Kết quả (val, 511 mẫu MLQA-vi)

| Cách làm | Jaccard | EM |
|---|---|---|
| Baseline `starter.py` (câu trùng nhiều từ nhất, lấy 5 từ đầu không thuộc câu hỏi) | 0.0948 | 0.0059 |
| XGBoost hồi quy Jaccard (train_vi) | 0.2230 | 0.0705 |
| XGBRanker (train_vi) | 0.2186 | 0.0568 |
| **XGBRanker (train_vi + 3000 mẫu SQuAD)** | **0.2270** | 0.0646 |
| Trần: span ứng viên tốt nhất trong 2 câu đã chọn | 0.8606 | - |

Đặc trưng quan trọng nhất: các cờ "năm nào", "khi nào", "bao nhiêu", "ai", số chữ số trong span, độ liên quan và hạng của câu. Chạy hoàn toàn trên CPU, khoảng 14 phút. Lần chạy này không có ghi chép điểm LB.

Khoảng cách lớn giữa 0.227 và trần 0.861 cho thấy bước chọn câu đã tốt, phần yếu là chọn đúng ranh giới span khi không có biểu diễn ngữ nghĩa.

## Phiên bản trước (dùng pretrained, code không còn giữ)

Trước khi chuyển sang luật không dùng pretrained, đội dùng mô hình đa ngữ đã học QA trên SQuAD, chạy zero-shot cross-lingual như các lời giải top của cuộc thi Kaggle chaii:

| Cách làm | val Jaccard | EM |
|---|---|---|
| `deepset/xlm-roberta-large-squad2` zero-shot | 0.651 | 0.489 |
| Ghép logit 0.6 x large + 0.4 x `deepset/xlm-roberta-base-squad2` fine-tune trên train_vi (3 epoch, đóng băng word embedding cho vừa 4 GB VRAM) | 0.663 | 0.495 |

Giải mã span: cửa sổ trượt 384 token, stride 128, 20 vị trí start/end tốt nhất mỗi cửa sổ, end ≥ start, dài tối đa 40 token. Riêng bản base sau fine-tune chỉ đạt 0.589 nhưng vẫn bổ sung được cho bản large.

## Chạy

Đặt `train_vi.json`, `validation_vi.json`, `test.json`, `train_en.json` cùng thư mục với notebook:

```
jupyter nbconvert --to notebook --execute solution.ipynb
```

Thư viện: `numpy`, `pandas`, `xgboost`.

## Tham khảo

- Rajpurkar, P. và cộng sự (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text*. [arXiv:1606.05250](https://arxiv.org/abs/1606.05250).
- Artetxe, M., Ruder, S., Yogatama, D. (2020). *On the Cross-lingual Transferability of Monolingual Representations* (XQuAD). [arXiv:1910.11856](https://arxiv.org/abs/1910.11856).
- Lewis, P. và cộng sự (2020). *MLQA: Evaluating Cross-lingual Extractive Question Answering*. [arXiv:1910.07475](https://arxiv.org/abs/1910.07475).
- Chen, T., Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754).
- Burges, C. J. C. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview*. Microsoft Research Technical Report MSR-TR-2010-82.
- Conneau, A. và cộng sự (2020). *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-R). [arXiv:1911.02116](https://arxiv.org/abs/1911.02116).
- Kaggle: [chaii - Hindi and Tamil Question Answering](https://www.kaggle.com/competitions/chaii-hindi-and-tamil-question-answering).
