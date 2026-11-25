# Sinh caption tiếng Việt cho ảnh (KTVIC)

Dạng bài **sinh mô tả ảnh** (đề chung kết OlpAI 2025 có bài này): viết một câu tiếng Việt mô tả nội dung mỗi ảnh. Ảnh là cảnh đời sống hằng ngày ở Việt Nam (đường phố, chợ, quán ăn, nhà cửa...).

## Overview

Mỗi ảnh train có 5 câu mô tả tiếng Việt do người viết, ví dụ:

```
đây là khung cảnh xuất hiện ở phía trước một căn nhà
có một căn nhà cao tầng xuất hiện ở trong bức ảnh
có một chiếc xe máy xuất hiện ở trong căn nhà
```

**Nhiệm vụ**: với mỗi ảnh test, sinh **một** câu mô tả.

**Luật (giống vòng khu vực OlpAI'26)**:

- Phần ảnh chỉ được dùng weights pretrained ImageNet trong danh sách của BTC, chỉ làm backbone:
  - torchvision (`weights="DEFAULT"`): resnet18, resnet34, resnet50, convnext_tiny, efficientnet_b0, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, regnet_y_400mf, densenet121, shufflenet_v2, squeezenet.
  - timm / Hugging Face: resnet18.a1_in1k, resnet50.a1_in1k, efficientnet_b0, mobilenetv3, microsoft/resnet.
- Không dùng model ngôn ngữ pretrained, không dùng model ảnh-chữ (CLIP, BLIP, SigLIP...), DINOv2, SAM. Phần sinh câu phải train từ đầu.
- Không dùng dữ liệu ngoài bộ được cấp (kể cả bản KTVIC gốc). Không viết caption tay cho ảnh test.
- 5 tiếng đầu: tối đa 20 lần nộp public. Tiếng cuối: mật khẩu `private_test.zip` được công bố, tối đa 5 lần nộp private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.

## Evaluation

**CIDEr-D** (cài đặt giống `pycocoevalcap`), so câu của bạn với 5 câu tham chiếu của ảnh:

- Tách từ: chữ thường, bỏ dấu câu, tách theo khoảng trắng (mỗi âm tiết là một từ).
- Mỗi câu thành vector TF-IDF của n-gram 1-4; IDF tính trên các câu tham chiếu của tập đang chấm, nên n-gram xuất hiện ở mọi ảnh ("trong bức ảnh") gần như không được điểm, còn n-gram đặc trưng ("chiếc xe máy", "quầy hàng") được điểm cao.
- Cosine giữa câu của bạn và từng câu tham chiếu, có cắt ngưỡng (lặp một từ nhiều lần không được thêm điểm) và phạt chênh độ dài `exp(-(Δ độ dài)^2 / 72)`; trung bình trên 4 bậc n-gram và 5 tham chiếu, nhân 10.

Thang điểm từ 0 trở lên; nộp một câu tham chiếu đúng của ảnh được khoảng 3,5, nộp cùng một câu chung chung cho mọi ảnh được khoảng 0,17.

## Submission File

`submission.csv` có tiêu đề `id,caption`, mỗi ảnh test một dòng (câu có dấu phẩy thì để trong dấu nháy kép, pandas tự làm):

```
id,caption
te012345,một người đàn ông đang bán hàng ở trên vỉa hè
```

Public là 279 ảnh trong `test/images/`, private là 279 ảnh trong `private_test.zip`. Nên nộp đủ 558 dòng; trước khi có mật khẩu, giữ nguyên các dòng private như `sample_submission.csv`.

## Data

```
train/images/*.jpg     3.769 ảnh
train_captions.json    {id: [5 câu mô tả]}
test/images/*.jpg      279 ảnh public
private_test.zip       (có mật khẩu) private_test/images/*.jpg: 279 ảnh private
sample_submission.csv  id,caption (cùng một câu cho mọi ảnh)
baseline.ipynb, baseline.py
LICENSE-DATASET.txt
```

Giải nén private test (khi có mật khẩu):

```python
import zipfile
zipfile.ZipFile("private_test.zip").extractall(".", pwd=b"<mật khẩu>")   # hoặc: unzip -P <mật khẩu> private_test.zip
```

## Hướng làm gợi ý

- **Truy hồi + đồng thuận** (baseline dùng đặc trưng màu thô): thay bằng đặc trưng ResNet / ConvNeXt ImageNet, tìm K ảnh train gần nhất, chọn câu đồng thuận nhất theo CIDEr giữa các câu ứng viên (MBR). Tăng K và chỉnh cách chấm đồng thuận.
- **Sinh câu**: đặc trưng ảnh ImageNet (đóng băng) + decoder LSTM / Transformer nhỏ train từ đầu, beam search; câu tiếng Việt ở đây khá rập khuôn nên mô hình nhỏ học được.
- Kết hợp: sinh nhiều câu rồi chọn bằng MBR.

Nguồn: KTVIC (Pham và cộng sự, 2024), bản Hugging Face `ai-enthusiasm-community/KTVIC`; test lấy từ tập test gốc (558 ảnh, 5 câu mỗi ảnh).
