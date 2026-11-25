# Bài 04 (4/12) - DimABSA tiếng Việt: hồi quy VA cho khía cạnh, trích bộ tứ

**Thời gian mục tiêu: ≤ 75 phút** (A 25, B 50). Đây là dạng đề NLP OlpAI'26.

## Đọc trước khi làm

Đọc trong khoảng 30-45 phút trước khi bấm giờ; ghi ra giấy 3-5 ý chính. Lúc làm bài thì đóng hết.

- **Bài học**: [on_tap/dang_bai/14_dimabsa.md](../../../on_tap/dang_bai/14_dimabsa.md) (đề thật, metric, hướng làm) và [10_gan_nhan_chuoi.md](../../../on_tap/dang_bai/10_gan_nhan_chuoi.md) (BIO, BiLSTM + ký tự).
- **Đề thật**: tài liệu AIO2026 `OAI2-Dimensional Aspect-Based Sentiment Analysis.pdf`, mục II (định dạng JSON, RMSE_VA, cF1).
- **scikit-learn**: [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).
- **PyTorch**: [nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) (`bidirectional`, `batch_first`), [cross_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) (`ignore_index`).
- **Bài báo**: Lample và cộng sự, *Neural Architectures for NER* ([arXiv:1603.01360](https://arxiv.org/abs/1603.01360)), mục 2-4.

Dữ liệu: review tiếng Việt giả lập (ứng dụng giao đồ ăn), dùng đúng 10 category của đề thật. Mỗi câu 1-3 bộ tứ. VA của opinion phụ thuộc từ gốc và từ bổ nghĩa: "rất / quá / siêu / cực kỳ / xỉu / dã man" đẩy xa trung tính và tăng arousal, "hơi / khá" kéo về trung tính, "không / chẳng" đảo chiều. Bẫy: teencode (`ko`, `k`, `qá`, `wá`), khoảng 30% khía cạnh trong test là từ **chưa có** trong train (`bún bò`, `túi giữ nhiệt`...), khía cạnh ẩn `NULL`.

## Phần A - Hồi quy VA cho khía cạnh cho trước (`bai_a.py`)

- **Đầu vào**: `data/a_train.jsonl` (`{"id","text","aspects":[{"aspect","valence","arousal"}]}`, 2.500 câu), `data/a_test.jsonl` (`{"id","text","aspects":[{"term"}]}`, 800 câu).
- **Đầu ra**: `pred_a.jsonl`, mỗi dòng `{"id", "aspects": [{"aspect", "valence", "arousal"}]}` đúng định dạng đề; `aspect` chép nguyên `term`; giá trị trong [1, 9], làm tròn 2 chữ số.
- **Metric**: RMSE_VA. **Ngưỡng ≤ 0.70** (lời giải mẫu 0.50; lấy cửa sổ ±4 từ quanh khía cạnh thay vì đúng mệnh đề của nó: 0.95; đoán trung bình: 2.6).
- **Khung**: chuẩn hoá NFC, chữ thường, teencode; lấy **mệnh đề chứa khía cạnh** (cắt tại dấu câu và "nhưng / mà / và / còn"), thay khía cạnh bằng `ASP`; TF-IDF từ 1-3 gram + ký tự; Ridge riêng V và A; kẹp [1, 9], làm tròn.

## Phần B - Trích bộ tứ (`bai_b.py`)

- **Đầu vào**: `data/b_train.jsonl` (`{"id","text","quadruplets":[{"aspect","category","opinion","valence","arousal"}]}`, 2.500 câu), `data/b_test.jsonl` (`{"id","text"}`, 800 câu).
- **Đầu ra**: `pred_b.jsonl`, mỗi dòng `{"id", "quadruplets": [...]}`; aspect / opinion chép đúng chuỗi trong câu (giữ chữ hoa, teencode), aspect ẩn ghi `"NULL"`.
- **Metric**: cF1 (khớp khi aspect, category, opinion trùng sau chữ thường; `cTP = 1 - khoảng cách VA / sqrt(128)`; bộ trùng lặp bị phạt). **Ngưỡng 0.70** (lời giải mẫu 0.79; tra từ điển aspect / opinion của train rồi ghép gần nhất: 0.57).
- **Khung**:
  1. Token hoá giữ vị trí ký tự (`re.finditer(r"\w+|[^\w\s]", text)`); nhãn BIO `B-ASP, I-ASP, B-OPI, I-OPI, O` từ bộ tứ train.
  2. BiLSTM + đặc trưng ký tự (CNN trên ký tự của từng từ) + **word dropout** (thay từ bằng `<unk>` 10%) để xử lý khía cạnh chưa gặp.
  3. Ghép mỗi opinion với aspect cùng mệnh đề gần nhất; không có aspect thì `NULL`.
  4. Category: LogisticRegression trên "aspect | opinion" (TF-IDF ký tự + từ). VA: Ridge trên chuỗi opinion (đã có từ bổ nghĩa).
  5. Khử trùng lặp, ghi JSON Lines.

## Bài 2 - Viết lại nhanh bài 03 (≤ 30 phút)

Trong `d03_hoi_dap`: `viet_lai.py`, chấm `python check_03.py viet_lai.py`.

## Cách chạy

```
python tao_du_lieu.py
python check_04.py bai_a.py bai_b.py
```

Lưu ý máy Windows: chương trình có `nn.LSTM` / `nn.GRU` đôi khi bị Windows báo lỗi `0xC0000409` lúc thoát dù đã chạy xong; bộ chấm chỉ cảnh báo và vẫn chấm file nộp.

## Mở rộng (không chấm)

- Thay Ridge ở phần A bằng BiGRU có đánh dấu vị trí khía cạnh và loss LogSigma (xem bài học 14).
- Ensemble 3 seed của bộ gán nhãn BIO (bỏ phiếu theo token).
- Chạy lời giải trên [selfhost_kaggle/DimABSA_Quad](../../../selfhost_kaggle/DimABSA_Quad/) (tiếng Anh, khó hơn nhiều).
