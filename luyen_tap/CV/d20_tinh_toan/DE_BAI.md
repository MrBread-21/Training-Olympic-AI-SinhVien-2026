# Bài 20 (20/11) - Tính toán cơ bản bằng numpy

**Thời gian mục tiêu: ≤ 60 phút cho 16 hàm.** Đóng tài liệu, không LLM, không tra mạng.

## Đọc trước khi làm

Đọc trong khoảng 30-45 phút trước khi bấm giờ; ghi ra giấy 3-5 ý chính (công thức, tham số, bẫy). Lúc làm bài thì đóng hết.

- **numpy**: [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) và [Indexing](https://numpy.org/doc/stable/user/basics.indexing.html) (phần advanced indexing: `a[np.arange(n), y]`), tham số `axis` / `keepdims`, `np.bincount`, `np.argsort`.
- **Softmax ổn định số học, cross-entropy**: CS231n [Linear classification](https://cs231n.github.io/linear-classify/), mục *Softmax classifier* và *Practical issues: Numeric stability*.
- **Metric phân loại**: scikit-learn [Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html), các mục confusion matrix, precision/recall/F1 (macro so với micro), ROC.
- **AUROC theo hạng**: Wikipedia [Mann-Whitney U test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test), mục *Area-under-curve (AUC) statistic for ROC curves*.
- **kNN**: scikit-learn [Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html), mục 1.6.2 (phân loại).
- **Kadane**: Wikipedia [Maximum subarray problem](https://en.wikipedia.org/wiki/Maximum_subarray_problem), mục *Kadane's algorithm*.
- **Trong máy**: `../../ON_TAP.md` phần 2.E (validation) và phần 3 mục 6 (metric tự viết).

## Đề

Viết thân 16 hàm trong `bai_tap.py` (đề từng hàm nằm ngay trong docstring). Chỉ dùng numpy và thư viện chuẩn; tránh vòng lặp Python khi numpy làm được.

| Nhóm | Bài |
|---|---|
| Chuẩn hoá dữ liệu | 01 mean/std theo kênh, 02 chuẩn hoá theo kênh, 03 one-hot |
| Softmax và loss | 04 softmax ổn định số học, 05 cross-entropy bằng log-sum-exp |
| Metric phân loại | 06 accuracy, 07 ma trận nhầm lẫn (không vòng lặp), 08 macro-F1, 09 AUROC theo hạng (có điểm bằng nhau) |
| Khoảng cách | 10 cosine, 11 Euclid bằng broadcasting, 12 kNN bỏ phiếu |
| Chuỗi và đoạn | 13 trung bình trượt mode edge, 14 tIoU đoạn khung (tính cả hai đầu), 15 Jaccard từ, 16 đoạn tổng lớn nhất (Kadane) |

## Chấm

```
python kiem_tra.py          # mọi bài
python kiem_tra.py 5 9      # chỉ bài 5 và 9
```

Bộ chấm so với sklearn / torch / cách tính trực tiếp, có các ca khó: logits cỡ 1000, điểm bằng nhau trong AUROC, lớp không xuất hiện trong macro-F1, mảng toàn số âm.

## Sau khi làm xong

Mở `loi_giai/loi_giai.py` để so cách viết. Ghi thời gian vào bảng tiến độ ở `../README.md`.
