# Bài 24 (24/11) - Truy xuất ảnh: GeM + ArcFace + kNN cosine

**Thời gian mục tiêu: ≤ 45 phút.**

## Đọc trước khi làm

Đọc trong khoảng 30-45 phút trước khi bấm giờ; ghi ra giấy 3-5 ý chính (công thức, tham số, bẫy). Lúc làm bài thì đóng hết.

- **Bài báo** (đọc phần phương pháp):
  - GeM: Radenović, Tolias, Chum, *Fine-tuning CNN Image Retrieval with No Human Annotation* [arXiv:1711.02512](https://arxiv.org/abs/1711.02512), mục 3.1 (công thức GeM, p học được).
  - ArcFace [arXiv:1801.07698](https://arxiv.org/abs/1801.07698), mục 2.1 (cộng margin vào góc, hệ số s).
  - BNNeck: Luo và cộng sự, *Bag of Tricks and A Strong Baseline for Deep Person Re-identification* [arXiv:1903.07071](https://arxiv.org/abs/1903.07071), mục 3 (BatchNorm trước lớp phân loại, embedding chuẩn hoá).
  - Musgrave và cộng sự, *A Metric Learning Reality Check* [arXiv:2003.08505](https://arxiv.org/abs/2003.08505), mục 2-3 (đánh giá đúng: tách lớp train / test, Recall@K, MAP@R).
- **PyTorch**: [torch.topk](https://pytorch.org/docs/stable/generated/torch.topk.html), [F.normalize](https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html).
- **Trong máy**: `../../../challenges/MSAD/main.py` (docstring có phần thử GeM, VLAD, kNN truy hồi).

## Bài 1 - Tìm ảnh cùng đối tượng

- **Đầu vào**: `data/train.csv` (`id,object`): 100 đối tượng x 6 ảnh (mỗi đối tượng là một tổ hợp hình chính có sọc + hình phụ + màu riêng; mỗi ảnh là một lần chụp khác nền, góc, cỡ). `data/gallery/` (120 ảnh) và `data/query/` (120 ảnh) gồm **60 đối tượng mới**, mỗi đối tượng 2 ảnh ở mỗi tập.
- **Đầu ra**: `submission.csv` (`id,matches`): với mỗi query, ít nhất 5 id gallery (cách nhau bởi dấu cách), xếp từ giống nhất.
- **Metric**: Recall@1, **ngưỡng 0.75** (in thêm Recall@5).
- **Khung bắt buộc**:
  1. Backbone ResNet-18 ImageNet, thay pooling bằng **GeM** (p học được), thêm lớp embedding (Linear + BatchNorm).
  2. Đầu **ArcFace** (s = 30, m = 0.3) trên nhãn đối tượng của train.
  3. Tách val **theo đối tượng** (20 đối tượng cuối), đo Recall@1 trên val sau mỗi epoch, lưu checkpoint tốt nhất.
  4. Embedding chuẩn hoá L2 (+ lật ngang làm TTA), xếp hạng gallery bằng cosine (`topk`).
- So sánh: GAP của ResNet-18 đóng băng không train chỉ được Recall@1 khoảng 0.22. Lời giải mẫu đạt 0.79-0.83 (Recall@5 khoảng 0.95, dao động giữa các lần chạy) trong khoảng 0.7 phút.

## Bài 2 - Viết lại nhanh bài 23 (≤ 20 phút)

Trong `d23_dac_trung_dong_bang`, viết `viet_lai.py`, chấm bằng `python check_23.py viet_lai.py`.

## Cách chạy

```
python tao_du_lieu.py
python check_24.py bai_lam.py
```
