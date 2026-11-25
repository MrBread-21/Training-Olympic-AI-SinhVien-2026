# CV5 - Phân loại đa nhãn: ảnh có những vật gì

## Overview

Mỗi ảnh có thể chứa một hoặc nhiều vật thuộc 20 lớp (người, chó, ô tô, ghế, chai...). Dự đoán tập lớp có trong ảnh.

Khó khăn chính là **chọn ngưỡng**: mô hình cho xác suất từng lớp, nhưng metric chấm theo nhãn cứng, và các lớp hiếm (chai, chậu cây, ghế) cần ngưỡng khác lớp phổ biến (người).

**Luật (giống vòng khu vực OlpAI'26)**:

- Chỉ dùng weights pretrained ImageNet trong danh sách sau, không tải model khác:
  - torchvision (`weights="DEFAULT"`): resnet18, resnet34, resnet50, convnext_tiny, efficientnet_b0, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, regnet_y_400mf, densenet121, shufflenet_v2, squeezenet.
  - timm / Hugging Face: resnet18.a1_in1k, resnet50.a1_in1k, efficientnet_b0, mobilenetv3, microsoft/resnet.
- Không dùng model nền tảng ảnh-chữ (CLIP, SigLIP), DINOv2, VideoMAE, SAM.
- Không dùng dữ liệu ngoài.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.

## Evaluation

**Macro-F1 đa nhãn**: với mỗi lớp tính F1 trên toàn bộ ảnh test (lớp đó có hay không có trong ảnh), rồi lấy trung bình 20 lớp.

Cột `labels`: tên lớp cách nhau bởi dấu cách, đúng chính tả như `classes.csv`. Được để trống nếu không có lớp nào.

```
id,labels
te123456,person dog
te654321,car
```

## Data

- `train/*.jpg`, `train.csv` (`id, labels`): 1.464 ảnh.
- `test/*.jpg`, `test.csv` (`id`): 1.449 ảnh.
- `classes.csv`: 20 lớp.
- `sample_submission.csv`, `baseline.py` (ResNet-18 đóng băng + 20 logistic, ngưỡng 0.5).

Ảnh đã thu về cạnh dài 384 px.

Nguồn: PASCAL VOC 2012 (Everingham và cộng sự), phần có mask phân đoạn; nhãn lấy từ các lớp xuất hiện trong mask.
