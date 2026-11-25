# Bài 02 (2/12) - Caption: truy hồi + đồng thuận, rồi decoder sinh câu

**Thời gian mục tiêu: ≤ 70 phút** (A 30, B 40). Mỗi lời giải mẫu chạy khoảng 4 phút trên GTX 1650.

## Đọc trước khi làm

Đọc trong khoảng 30-45 phút trước khi bấm giờ; ghi ra giấy 3-5 ý chính. Lúc làm bài thì đóng hết.

- **Bài học**: [on_tap/dang_bai/07_caption.md](../../../on_tap/dang_bai/07_caption.md) và metric CIDEr-D trong [on_tap/01_metric.md](../../../on_tap/01_metric.md).
- **Bài báo**:
  - Vedantam và cộng sự, *CIDEr* ([arXiv:1411.5726](https://arxiv.org/abs/1411.5726)), mục 3 (TF-IDF n-gram, cosine, trung bình các tham chiếu).
  - Devlin và cộng sự, *Exploring Nearest Neighbor Approaches for Image Captioning* ([arXiv:1505.04467](https://arxiv.org/abs/1505.04467)), mục 2 (chọn câu đồng thuận).
  - Vinyals và cộng sự, *Show and Tell* ([arXiv:1411.4555](https://arxiv.org/abs/1411.4555)), mục 3 (CNN + RNN, teacher forcing).
- **PyTorch**: [nn.GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) (`h0` có shape `(num_layers, B, H)`), [cross_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) (`ignore_index`).
- **Trong máy**: [challenges/ImgCap](../../../challenges/ImgCap/) README (học "từ khái niệm" + truy hồi + MBR).

## Đề

- **Đầu vào**: `data/train/*.png` + `data/train_captions.json` (`{id: [5 câu]}`, 3.000 ảnh 64x64), `data/test/*.png` (600 ảnh).
- Mỗi ảnh có 1-2 vật: hình (tròn, vuông, tam giác, sao), màu (đỏ, xanh lá, xanh dương, vàng, tím, trắng), cỡ (nhỏ, lớn), vị trí (bên trái, bên phải, phía trên, phía dưới, ở giữa). Câu mô tả theo 5 kiểu diễn đạt.
- **Đầu ra**: `submission_a.csv`, `submission_b.csv` (`id,caption`), mỗi ảnh test một câu.
- **Metric**: CIDEr-D (n-gram 1-4, IDF trên tham chiếu của tập test, phạt chênh độ dài, nhân 10).

## Phần A - Truy hồi + đồng thuận (`bai_a.py`)

- **Ngưỡng 3.6** (lời giải mẫu 4.27; truy hồi bằng **pixel thu nhỏ** chỉ 2.63).
- **Khung**: train CNN nhỏ từ đầu dự đoán "từ khái niệm" (multi-label BCE: hình, màu, cỡ, vị trí có trong 5 câu của ảnh); đặc trưng = lớp trước head, **giữ vị trí** (đừng pool toàn cục), chuẩn hoá L2; K ảnh train gần nhất; gom 5K câu; chọn câu có tổng tương đồng TF-IDF n-gram (có trọng số theo độ giống ảnh) lớn nhất với các câu khác.

## Phần B - Decoder sinh câu (`bai_b.py`)

- **Ngưỡng 4.5**: phải **thắng truy hồi** (lời giải mẫu 4.81). Dữ liệu tổ hợp (2 vật x 6 màu x 4 hình x 5 vị trí...) nên ảnh test hiếm khi có ảnh train giống hẳn: truy hồi mượn câu của ảnh "gần giống" và sai một chi tiết, còn decoder ghép được tổ hợp mới.
- **Khung**: vocab từ câu train (`<pad>`, `<bos>`, `<eos>`); encoder CNN thành vector d; GRU với `h0` = vector ảnh và vector ảnh ghép vào mỗi bước; teacher forcing, mỗi epoch chọn ngẫu nhiên 1 trong 5 câu; `cross_entropy(..., ignore_index=PAD, label_smoothing=0.1)`; giải mã greedy tới `<eos>`. **Không lật ảnh** khi augmentation (đổi trái / phải làm sai câu).
- Train đủ lâu: 25 epoch chỉ được 3.96 (thua truy hồi), 40 epoch được 4.81.

## Bài 2 - Viết lại nhanh bài 01 (≤ 30 phút)

Trong `d01_am_thanh`: `viet_lai.py`, chấm `python check_01.py viet_lai.py`.

## Cách chạy

```
python tao_du_lieu.py
python check_02.py bai_a.py bai_b.py
```

## Mở rộng (không chấm)

- Beam search 3 + chặn lặp trigram cho phần B.
- Sinh 5 câu bằng beam rồi chọn bằng MBR cùng với các câu truy hồi.
- Chạy phần A với đặc trưng ResNet ImageNet trên [selfhost_kaggle/KTVIC_Captioning](../../../selfhost_kaggle/KTVIC_Captioning/) (ảnh thật, tiếng Việt).
