# cv3_bat_thuong - nội dung từng ô trên Kaggle

## Title

OlpAI thử - CV3 Phát hiện sản phẩm lỗi

## Competition URL

olpai-thu5-bat-thuong

## Subtitle

Chấm điểm bất thường cho 5 loại sản phẩm khi train chỉ có ảnh tốt. Chỉ backbone ImageNet.

## Overview

Phát hiện sản phẩm lỗi trên dây chuyền khi tập train chỉ có ảnh sản phẩm tốt, chấm bằng AUROC chung cho 5 loại. Chỉ dùng backbone ImageNet trong danh sách của BTC.

## Description

Dây chuyền sản xuất 5 loại sản phẩm: chai (bottle), thảm (carpet), hạt dẻ (hazelnut), đai ốc (metal_nut), ốc vít (screw). Tập train **chỉ có ảnh sản phẩm tốt**. Tập test có cả sản phẩm tốt và sản phẩm lỗi (vết xước, nứt, dính bẩn, thiếu chi tiết, cong...). Loại lỗi trong test không xuất hiện ở train.

Với mỗi ảnh test, cho một điểm bất thường: càng lớn càng có khả năng là lỗi.

## Evaluation

**AUROC** tính chung trên toàn bộ ảnh test của cả 5 loại. Vì tính chung, điểm của các loại phải **so sánh được với nhau**: nếu loại này có điểm cao hơn loại kia một cách có hệ thống thì AUROC sẽ giảm, nên cần chuẩn hoá điểm theo từng loại.

```
id,score
te123456,0.8731
te654321,-1.204
```

## Timeline

- Bắt đầu: thứ 5, 26/11/2026 (giờ bắt đầu do người tổ chức thông báo).
- Kết thúc: 6 tiếng sau giờ bắt đầu. 5 tiếng đầu làm bài và nộp; tiếng cuối chốt mô hình, viết `main.py`, chọn 2 bài nộp cuối.
- Bảng private được mở ngay sau khi kết thúc.

## Rules

- Chỉ dùng weights pretrained ImageNet trong danh sách sau, không tải model khác:
  - torchvision (`weights="DEFAULT"`): resnet18, resnet34, resnet50, convnext_tiny, efficientnet_b0, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, regnet_y_400mf, densenet121, shufflenet_v2, squeezenet.
  - timm / Hugging Face: resnet18.a1_in1k, resnet50.a1_in1k, efficientnet_b0, mobilenetv3, microsoft/resnet.
- Không dùng model nền tảng ảnh-chữ (CLIP, SigLIP), DINOv2, VideoMAE, SAM.
- Không dùng dữ liệu ngoài.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Không dùng bản MVTec AD gốc.
- Không gán nhãn tay cho test.
- Tối đa 20 lần nộp; chọn 2 bài nộp cuối để chấm private.

## Citation

MVTec AD (Bergmann và cộng sự, CVPR 2019), CC BY-NC-SA 4.0.

## Dataset Description

- `train/*.png`, `train.csv` (`id, category`): 1.420 ảnh sản phẩm tốt.
- `test/*.png`, `test.csv` (`id, category`): 585 ảnh, khoảng 74% là lỗi.
- `sample_submission.csv`, `baseline.py` (chạy `python baseline.py /kaggle/input/olpai-thu5-bat-thuong`) (đặc trưng ResNet-18 toàn ảnh, khoảng cách tới ảnh tốt gần nhất).

Ảnh đã thu về 256x256.

## Data license

CC BY-NC-SA 4.0

## Files để tải lên

- Data: `cv3_bat_thuong_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py`
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Kiểm tra bằng sandbox submission: nộp file từ `baseline.py`, điểm phải gần `host/min_score.txt`
