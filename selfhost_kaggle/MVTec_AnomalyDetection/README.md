# CV3 - Phát hiện sản phẩm lỗi khi chỉ có ảnh sản phẩm tốt

## Overview

Dây chuyền sản xuất 5 loại sản phẩm: chai (bottle), thảm (carpet), hạt dẻ (hazelnut), đai ốc (metal_nut), ốc vít (screw). Tập train **chỉ có ảnh sản phẩm tốt**. Tập test có cả sản phẩm tốt và sản phẩm lỗi (vết xước, nứt, dính bẩn, thiếu chi tiết, cong...). Loại lỗi trong test không xuất hiện ở train.

Với mỗi ảnh test, cho một điểm bất thường: càng lớn càng có khả năng là lỗi.

**Luật (giống vòng khu vực OlpAI'26)**:

- Chỉ dùng weights pretrained ImageNet trong danh sách sau, không tải model khác:
  - torchvision (`weights="DEFAULT"`): resnet18, resnet34, resnet50, convnext_tiny, efficientnet_b0, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, regnet_y_400mf, densenet121, shufflenet_v2, squeezenet.
  - timm / Hugging Face: resnet18.a1_in1k, resnet50.a1_in1k, efficientnet_b0, mobilenetv3, microsoft/resnet.
- Không dùng model nền tảng ảnh-chữ (CLIP, SigLIP), DINOv2, VideoMAE, SAM.
- Không dùng dữ liệu ngoài.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Không dùng bản MVTec AD gốc.

## Evaluation

**AUROC** tính chung trên toàn bộ ảnh test của cả 5 loại. Vì tính chung, điểm của các loại phải **so sánh được với nhau**: nếu loại này có điểm cao hơn loại kia một cách có hệ thống thì AUROC sẽ giảm, nên cần chuẩn hoá điểm theo từng loại.

```
id,score
te123456,0.8731
te654321,-1.204
```

## Data

- `train/*.png`, `train.csv` (`id, category`): 1.420 ảnh sản phẩm tốt.
- `test/*.png`, `test.csv` (`id, category`): 585 ảnh, khoảng 74% là lỗi.
- `sample_submission.csv`, `baseline.py` (đặc trưng ResNet-18 toàn ảnh, khoảng cách tới ảnh tốt gần nhất).

Ảnh đã thu về 256x256.

Nguồn: MVTec AD (Bergmann và cộng sự, CVPR 2019), CC BY-NC-SA 4.0.
