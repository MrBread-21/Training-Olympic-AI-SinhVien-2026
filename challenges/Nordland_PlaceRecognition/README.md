# Lời giải CV0 - Nordland: nhận dạng địa điểm qua bốn mùa

Đề: [selfhost_kaggle/Nordland_PlaceRecognition](../../selfhost_kaggle/Nordland_PlaceRecognition/). Code: [train_fast.py](train_fast.py). Log lần chạy chính: [train_fast.log](train_fast.log).

## Kết quả

| | Val R@1 | Val R@5 | Val R@10 |
|---|---|---|---|
| Baseline pixel (theo đề, điểm public) | ~0.07 | | |
| ResNet-50 ImageNet dùng nguyên (theo đề, điểm public) | ~0.03 | | |
| **Lời giải này, train 25 phút** | **0.929** | 0.978 | 0.987 |

Trên test (chấm bằng đúng metric của cuộc thi): **public 0.814, private 0.835**.

- Val là 15% cuối của train, tách theo đoạn hành trình giống cách đề chia test. Test thấp hơn val khoảng 0.1: đoạn test là 30% cuối hành trình, xa đoạn train hơn đoạn val, cảnh vật khác hơn (val chỉ cách train 50 khung).
- Máy: GTX 1650 4GB, fp32. Tổng thời gian chạy 25.6 phút (nạp dữ liệu 7 giây, train 25 phút, val và dự đoán test khoảng 30 giây).
- Từ phút 15 val đã chững ở khoảng 0.93, nên train 15 phút là đủ.

## Ý tưởng trong một câu

Train ResNet18 ImageNet biến mỗi ảnh thành một vector **giữ bố cục không gian** (lưới 4x7 từ layer3), bằng loss contrastive giữa ảnh đông và ảnh hè cùng vị trí, với batch gồm **các khung liền kề nhau** làm negative khó. Lúc dự đoán chỉ cần tìm ảnh hè có vector gần nhất.

## Ba điều quyết định điểm số

Mỗi điều dưới đây là kết quả của một lần chạy thất bại.

### 1. Descriptor phải giữ bố cục không gian

Bản đầu dùng layer4 + GeM + Linear, tức gộp cả ảnh thành một vector toàn cục. Loss train giảm về 0.045, nhưng **val R@1 cũng chỉ 0.045**, thấp hơn cả baseline pixel.

Lý do: camera gắn cố định ở đầu tàu, nên ở mọi mùa đường ray luôn nằm giữa phía dưới, trời ở trên, cột điện và cây ở hai bên. Thông tin "vật gì nằm ở ô nào" là manh mối mạnh nhất, và gộp toàn ảnh làm mất nó. Baseline pixel 64x32 thắng được ResNet-50 cũng vì lý do này.

Sửa: lấy feature map của layer3 (256x8x14), giảm kênh bằng conv 1x1 xuống 128, avg pool về lưới 4x7 rồi làm phẳng thành 3584 chiều.

### 2. Negative phải là các khung liền kề

Nếu batch gồm các vị trí ngẫu nhiên khắp 500 km đường ray, phân biệt chúng quá dễ: loss về gần 0 mà model không học được gì có ích. Trong khi đó, bài toán thật là phân biệt khung i với khung i+3, hai ảnh gần như giống hệt nhau.

Sửa: mỗi batch 64 cặp gồm khoảng 8 cụm, mỗi cụm 8 khung liên tiếp cách nhau 3-6 khung. Cặp lệch ≤ 2 khung bị loại khỏi negative, vì luật chấm tính chúng là đúng.

Hai thay đổi 1 và 2 cùng lúc đưa val R@1 **từ 0.045 lên 0.897 chỉ sau 3 phút train**.

### 3. Đóng băng toàn bộ BatchNorm

Khi kéo dài lên 25 phút, val ở phút thứ 5 **sụp còn 0.0035**, trong khi loss train vẫn thấp.

Lý do: mỗi batch toàn các khung gần giống nhau, nên thống kê của batch (mean, var) lệch rất xa running stats. Ở chế độ train, model chuẩn hóa theo thống kê của batch; khi eval thì dùng running stats. Model học một kiểu nhưng bị đánh giá theo kiểu khác.

Sửa: ghi đè `Net.train()` để mọi `BatchNorm2d` luôn ở chế độ eval, dùng thống kê ImageNet có sẵn; tham số affine (gamma, beta) vẫn được train. Sau khi sửa, val tăng đều và không sụp nữa.

## Chi tiết từng phần

### Dữ liệu và chia val

- Sắp 19.289 cặp ảnh theo `frame`. Train trên 16.345 khung đầu, bỏ trống 50 khung, val trên 2.894 khung cuối (85% / 15%).
- Val được tách theo đoạn hành trình, giống cách đề tách test, nên điểm val phản ánh đúng khả năng gặp đoạn đường chưa thấy.

### Nạp ảnh nhanh

Ổ F: của máy nhà đọc file ảnh lẻ mất 10-13 ms/ảnh, nên 55k ảnh tốn khoảng 9 phút chỉ để đọc. Cách xử lý:

- Đọc **byte JPEG gốc** của cả 55k ảnh từ `cv0_nordland_data.zip` vào RAM: 7 giây, khoảng 750MB. Không giải nén trước, vì RAM trống chỉ khoảng 3GB.
- Mỗi bước train, giải nén batch trên GPU bằng `torchvision.io.decode_jpeg(..., device="cuda")` (khoảng 1900 ảnh/s), rồi `F.interpolate` từ 320x180 xuống 224x128.
- Không dùng DataLoader; worker trên Windows vừa khởi động chậm vừa nhân bản bộ nhớ.
- Không có zip (ví dụ trên Kaggle) thì script tự đọc từ thư mục.

### Augment (trên GPU)

| Loại | Cách làm | Áp dụng |
|---|---|---|
| Màu | độ sáng ±30%, độ tương phản ±30%, độ bão hòa ±40%, 20% thành ảnh xám | riêng từng ảnh |
| Hình học | phóng 88-100%, dịch ngang ±6%, dịch dọc ±4% | riêng từng ảnh |
| Lật ngang | xác suất 50% | chung cho cả cặp đông-hè |

Augment màu giúp model không dựa vào màu, vì tuyết trắng và cây xanh khác nhau hoàn toàn. Augment hình học mô phỏng việc tàu dừng lệch vị trí đôi chút giữa các mùa. Lật ngang chung cho cả cặp để hai ảnh vẫn khớp nhau.

### Model

```
ảnh 3x128x224
  - ResNet18 ImageNet: conv1, layer1, layer2, layer3   - 256x8x14
  - conv 1x1                                          - 128x8x14
  - adaptive avg pool                                 - 128x4x7
  - làm phẳng + chuẩn hóa L2                          - vector 3584 chiều
```

- Đóng băng trọng số conv1, bn1 và layer1: các tầng này chỉ bắt cạnh, góc, nên đóng băng vừa nhanh hơn vừa ít overfit.
- Bỏ layer4: ở độ phân giải này nó chỉ còn 4x7 ô và quá trừu tượng; bỏ đi còn nhanh hơn khoảng 30%.
- Weights: `torchvision.models.resnet18(weights="DEFAULT")`, nằm trong danh sách của BTC.

### Loss

InfoNCE hai chiều:

1. Tính ma trận tương đồng 64x64 giữa ảnh đông và ảnh hè, chia cho `tau` = 0.05.
2. Gán -inf cho các ô ngoài đường chéo có hai khung lệch ≤ 2.
3. Cross-entropy theo hàng (đông tìm hè) và theo cột (hè tìm đông), rồi lấy trung bình.

### Lịch train

- AdamW, lr 3e-4, weight decay 1e-4, batch 64 cặp (128 ảnh mỗi lần forward), fp32. GTX 1650 không có tensor core nên AMP còn chậm hơn.
- **Lịch learning rate theo thời gian, không theo epoch**: tăng dần trong 3% thời gian đầu, sau đó giảm theo cosine về 0 đúng lúc hết `--minutes`. Nhờ vậy, đổi thời gian train không cần chỉnh thêm gì.
- Run 25 phút: 4.517 bước, khoảng 385 ảnh/s, mỗi cặp train được thấy khoảng 18 lần.

### Dự đoán

- TTA: lấy tổng vector của ảnh gốc và ảnh lật ngang, rồi chuẩn hóa lại.
- Nhân ma trận vector query (8253) với vector database (8253), lấy top 10 cho mỗi query, ghi `id,predictions`.

## Diễn biến val của run chính

| Phút | 1.5 | 3 | 4.5 | 7.5 | 10.5 | 12 | 15 | 18 | 21 | 25 (cuối) |
|---|---|---|---|---|---|---|---|---|---|---|
| R@1 | 0.889 | 0.881 | 0.907 | 0.901 | 0.913 | 0.918 | **0.933** | 0.932 | 0.930 | 0.929 |

## Nhật ký thử nghiệm

| # | Thay đổi | Thời gian | Val R@1 |
|---|---|---|---|
| 1 | layer4 + GeM, batch ngẫu nhiên + negative cách 5-20 khung | 2 phút | 0.045 |
| 2 | lưới 4x7 từ layer3 + batch theo cụm khung liền kề | 3 phút | 0.897 |
| 3 | như #2, kéo lên 25 phút | dừng ở 6.5 phút | 0.0035 ở phút 5 (sụp) |
| 4 | như #3 + đóng băng BatchNorm | 25 phút | **0.929** |

Benchmark tốc độ train thuần trên GTX 1650 (ảnh/s, fp32):

| | 128x224 | 160x288 |
|---|---|---|
| resnet18 | 392 | 224 |
| resnet34 | 224 | 125 |
| resnet50 | 108 | 69 |

## Cách chạy

Máy nhà:

```
python train_fast.py                       # train 85%, val 15%, in R@1/5/10, ghi submission.csv
python train_fast.py --full --minutes 15   # train trên toàn bộ train, dùng cho bài nộp cuối
```

Kaggle (bật GPU và Internet để tải weights ImageNet):

```
%%writefile train_fast.py
<dán nội dung train_fast.py>
```

```
!python train_fast.py --data /kaggle/input/competitions/pka-olpai-test-nordland --minutes 15 --full
```

- Phải chạy qua `!python`, không chạy thẳng trong cell: `argparse` sẽ lỗi vì nhận tham số của Jupyter.
- Nếu dữ liệu bị lồng thêm một thư mục con, sửa `--data` cho trỏ tới thư mục chứa `train.csv`.

Tham số chính: `--minutes`, `--arch` (resnet18/resnet34/resnet50), `--head` (grid/gem), `--batch`, `--lr`, `--tau`, `--eval_every`, `--seed`, `--out`.

## Lỗi môi trường đã gặp trên máy nhà

- `import torch` phải đứng **trước** numpy/pandas, nếu không sẽ gặp `WinError 1114` khi nạp `c10.dll`.
- Console Windows không in được tiếng Việt: chạy kèm `PYTHONIOENCODING=utf-8`.
- Nên kiểm tra `nvidia-smi` trước khi train: nếu tiến trình khác đang chiếm VRAM, tốc độ sẽ giảm.

## Chưa làm, có thể thử tiếp

- **Chưa kiểm chứng:** `--full`, chạy trên Kaggle. Val nên tách xa train hơn (ví dụ bỏ trống vài nghìn khung) để sát với khoảng cách train-test.
- `main.py` theo luật (chạy lại dưới 20 phút, seed cố định): chỉ cần `--full --minutes 15`.
- resnet34 (chậm hơn khoảng 1.7 lần) hoặc ảnh lớn hơn 160x288.
- Lưới mịn hơn (ví dụ 8x14) hoặc thêm feature của layer2.
- Re-ranking: so khớp lại top 10 bằng feature map đầy đủ, hoặc tận dụng việc các ảnh database gần nhau cũng là các vị trí gần nhau.
