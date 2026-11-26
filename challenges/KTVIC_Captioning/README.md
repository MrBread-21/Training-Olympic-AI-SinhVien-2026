# Lời giải KTVIC_Captioning - Sinh caption tiếng Việt cho ảnh

Đề: [selfhost_kaggle/KTVIC_Captioning](../../selfhost_kaggle/KTVIC_Captioning/). Code: [solve.py](solve.py). Phần ảnh dùng backbone ImageNet trong danh sách BTC (đóng băng); phần ngôn ngữ không dùng model nào.

## Kết quả (chấm bằng đúng metric của cuộc thi, `selfhost_kaggle/tools/cham.py`)

| | Public | Private |
|---|---|---|
| Baseline của đề (ảnh thu nhỏ + histogram màu, K = 10) | 0,196 | 0,217 |
| **Lời giải này** | **1,029** | **0,925** |

Val (400 ảnh train để riêng): CIDEr-D theo K: 5: 0,78; 15: 0,93; 30: 0,98; 60: 0,98. Thời gian: 14,5 phút trên GTX 1650 (đặc trưng 3,5 phút; phần lớn còn lại là MBR viết bằng Python thuần).

## Cách làm

1. **Đặc trưng ảnh**: ResNet-50 (`weights="DEFAULT"`, bỏ `fc`) và ConvNeXt-Tiny (`weights="DEFAULT"`, bỏ lớp cuối), ảnh 224 x 224, TTA lật ngang; chuẩn hoá L2 từng mô hình rồi nối (2.816 chiều).
2. **Truy hồi**: cosine với 3.769 ảnh train, lấy K ảnh gần nhất (K chọn trên val: 60).
3. **Chọn câu đồng thuận (MBR)**: gom 5K câu của các láng giềng; điểm mỗi câu = tổng cosine TF-IDF n-gram 1-4 với các câu còn lại, có trọng số `độ_giống_ảnh^3` (láng giềng gần được ưu tiên). IDF tính trên caption train.

## Ghi chú

- Câu tiếng Việt của KTVIC khá rập khuôn ("có một ... đang ... ở trong bức ảnh"), nên truy hồi + đồng thuận rất mạnh; đặc trưng ảnh tốt là phần quyết định (baseline màu thô chỉ 0,2).
- K lớn có lợi (giống bài ImgCap): MBR trên nhiều ứng viên chọn được câu "trung bình" an toàn theo CIDEr.
- Hướng tăng tiếp: decoder LSTM / Transformer nhỏ trên đặc trưng lưới 7 x 7 (đóng băng) sinh câu, rồi MBR chung với câu truy hồi; viết MBR bằng ma trận thưa để nhanh hơn 10 lần.

## Chạy lại

```
python solve.py <thư_mục upload của đề>     # giải nén private_test.zip vào thư mục làm việc trước nếu muốn dự đoán cả private
python ../../selfhost_kaggle/tools/cham.py KTVIC_Captioning submission.csv
```
