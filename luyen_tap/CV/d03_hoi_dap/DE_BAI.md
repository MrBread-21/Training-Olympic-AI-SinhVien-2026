# Bài 03 (3/12) - Hỏi đáp trích span: chọn câu, sinh ứng viên, xếp hạng, ngưỡng để trống

**Thời gian mục tiêu: ≤ 60 phút.** Không cần GPU (lời giải mẫu 1,5 phút trên CPU).

## Đọc trước khi làm

Đọc trong khoảng 30-45 phút trước khi bấm giờ; ghi ra giấy 3-5 ý chính. Lúc làm bài thì đóng hết.

- **Bài học**: [on_tap/dang_bai/11_hoi_dap_trich_xuat.md](../../../on_tap/dang_bai/11_hoi_dap_trich_xuat.md).
- **scikit-learn**: [DictVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) (đặc trưng dạng dict thành ma trận thưa), [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).
- **XGBoost**: [XGBRegressor / XGBRanker](https://xgboost.readthedocs.io/en/stable/python/python_api.html) (`tree_method="hist"`, tham số `group` của ranker).
- **Bài báo**: Rajpurkar và cộng sự, *Know What You Don't Know* (SQuAD 2.0, [arXiv:1806.03822](https://arxiv.org/abs/1806.03822)), mục 2-3 (câu hỏi không đáp án, F1 theo từ).
- **Trong máy**: [challenges/VietnameseExtract](../../../challenges/VietnameseExtract/) README (37 đặc trưng span, XGBRanker).

## Đề

- **Đầu vào**: `data/train.csv` (`id,context_id,context,question,answer,answer_start`; 3.500 câu hỏi trên 700 đoạn tiểu sử nhân vật hư cấu; `answer` rỗng và `answer_start = -1` khi không có đáp án), `data/test.csv` (`id,context_id,context,question`; 1.300 câu hỏi trên 260 đoạn **khác**).
- **Đầu ra**: `submission.csv` (`id,answer`): chuỗi chép từ đoạn văn, hoặc để trống.
- **Metric**: F1 theo từ (chữ thường, NFC, tách `\w+`); câu không đáp án để trống được 1. **Ngưỡng 0.80**.
  - Để trống hết: 0.20. Trả cả câu chứa nhiều từ của câu hỏi nhất: 0.12. Lời giải mẫu **quên chuẩn hoá NFC**: 0.60.

Dữ liệu có bẫy:

- Mỗi đoạn có nhiều năm (sinh, tốt nghiệp, ra mắt tác phẩm, nhận giải) và nhiều con số (công trình, học trò); câu hỏi diễn đạt nhiều cách ("sinh năm nào", "ra đời vào năm nào", "năm sinh của ... là năm nào").
- Khoảng 22% câu hỏi hỏi về thông tin **không có** trong đoạn (đoạn văn bỏ ngẫu nhiên một số câu).
- 40% đoạn văn và câu hỏi của test ở dạng Unicode **NFD**, train toàn NFC.

## Khung cần viết (`bai_lam.py`)

1. NFC cho đoạn văn và câu hỏi.
2. **Chọn câu**: tách câu, điểm câu = số từ viết thường của câu hỏi có trong câu (bỏ tên riêng); lấy 2 câu cao nhất.
3. **Sinh span** 1-6 token trong 2 câu đó; bỏ span gồm toàn từ của câu hỏi. Chuỗi span lấy theo **vị trí ký tự gốc** trong đoạn văn.
4. **Đặc trưng span**: độ dài, là số / là năm 4 chữ số, chữ hoa đầu, toàn chữ hoa đầu, số từ câu hỏi trong span, khoảng cách tới từ câu hỏi gần nhất, nằm trong ngoặc kép, từ bên trái 1-2 vị trí, từ bên phải, cờ loại câu hỏi ("năm nào", "ở đâu", "bao nhiêu", "tốt nghiệp", "trao"...), và tương tác "loại câu hỏi x từ bên trái".
5. Nhãn = F1 của span với đáp án (0 nếu không có đáp án); `XGBRegressor` (hoặc `XGBRanker` nhóm theo câu hỏi).
6. Val chia **theo đoạn văn** (`GroupShuffleSplit` theo `context_id`); chọn **ngưỡng để trống** trên val: điểm span tốt nhất dưới ngưỡng thì trả rỗng.

## Bài 2 - Viết lại nhanh bài 02 phần A (≤ 25 phút)

Trong `d02_caption`: `viet_lai_a.py` (truy hồi + MBR), chấm `python check_02.py viet_lai_a.py`.

## Cách chạy

```
python tao_du_lieu.py
python check_03.py bai_lam.py
```

## Mở rộng (không chấm)

- So `XGBRegressor` với `XGBRanker(objective="rank:pairwise")` (nhãn F1 quy về 0-4, `group` = số span của từng câu hỏi theo thứ tự).
- Thêm đặc trưng "chênh lệch điểm span tốt nhất và nhì" cho quyết định để trống.
- Chạy lời giải trên [selfhost_kaggle/ViQuAD_ExtractiveQA](../../../selfhost_kaggle/ViQuAD_ExtractiveQA/) (dữ liệu thật, khó hơn nhiều).
