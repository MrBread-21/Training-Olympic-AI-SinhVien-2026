# CV1 - Phân loại họ máy bay khi nhãn nhiễu và ảnh test bị làm hỏng

## Overview

Phân loại ảnh máy bay vào 70 họ (Boeing 737, A320, ATR-72, Cessna 172...). Đây là bài phân loại chi tiết: các lớp rất giống nhau, chỉ khác ở số cửa sổ, hình đuôi, vị trí động cơ.

Hai khó khăn cố ý:

- **Khoảng 15% nhãn train bị sai**, thường bị gán nhầm sang họ khác cùng hãng (ví dụ A330 thành A340).
- **Khoảng 60% ảnh test bị làm hỏng**: nén JPEG nặng, mờ, nhiễu, đổi sáng tối, bị che một mảng. Ảnh train thì sạch.

**Luật (giống vòng khu vực OlpAI'26)**:

- Chỉ dùng weights pretrained ImageNet trong danh sách sau, không tải model khác:
  - torchvision (`weights="DEFAULT"`): resnet18, resnet34, resnet50, convnext_tiny, efficientnet_b0, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, regnet_y_400mf, densenet121, shufflenet_v2, squeezenet.
  - timm / Hugging Face: resnet18.a1_in1k, resnet50.a1_in1k, efficientnet_b0, mobilenetv3, microsoft/resnet.
- Không dùng model nền tảng ảnh-chữ (CLIP, SigLIP), DINOv2, VideoMAE, SAM.
- Không dùng dữ liệu ngoài.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.

## Evaluation

**Macro-F1** trên 70 lớp.

```
id,label
te123456,Boeing 737
te654321,A320
```

## Data

- `train/*.jpg` và `train.csv` (`id, label`): 3.334 ảnh, nhãn có nhiễu.
- `test/*.jpg` và `test.csv` (`id`): 3.333 ảnh.
- `classes.csv`: tên 70 họ, `label` trong file nộp phải đúng chính tả như ở đây.
- `sample_submission.csv`, `baseline.py` (đặc trưng ResNet-18 đóng băng + logistic).

Ảnh đã bỏ dải chữ bản quyền ở đáy và thu về cạnh dài 400 px.

Nguồn: FGVC-Aircraft (Maji và cộng sự, 2013), chỉ dùng cho nghiên cứu phi thương mại.
