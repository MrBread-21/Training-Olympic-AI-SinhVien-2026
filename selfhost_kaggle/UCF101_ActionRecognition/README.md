# CV4 - Nhận dạng hành động trong video

## Overview

Phân loại clip video ngắn vào 10 hành động (ném bóng chày, úp rổ, chơi bi-a, em bé bò...). Dữ liệu ít: chưa tới 300 clip train.

**Bẫy cần chú ý**: trong train, mỗi nhóm `gXX` (tên file `v_<Lớp>_gXX_cYY.avi`) là các clip cắt từ **cùng một video gốc**, nên rất giống nhau. Test gồm các nhóm hoàn toàn khác train. Chia val ngẫu nhiên theo clip sẽ cho điểm val ảo cao; phải chia theo nhóm.

**Luật (giống vòng khu vực OlpAI'26)**:

- Chỉ dùng weights pretrained ImageNet trong danh sách sau, không tải model khác:
  - torchvision (`weights="DEFAULT"`): resnet18, resnet34, resnet50, convnext_tiny, efficientnet_b0, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, regnet_y_400mf, densenet121, shufflenet_v2, squeezenet.
  - timm / Hugging Face: resnet18.a1_in1k, resnet50.a1_in1k, efficientnet_b0, mobilenetv3, microsoft/resnet.
- Không dùng model nền tảng ảnh-chữ (CLIP, SigLIP), DINOv2, VideoMAE, SAM.
- Không dùng dữ liệu ngoài.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Không dùng model video huấn luyện trên Kinetics (r3d_18, x3d...): chỉ 2D CNN ImageNet trên từng khung rồi gộp theo thời gian.

## Evaluation

**Macro-F1** trên 10 lớp.

```
id,label
v123456,BasketballDunk
```

## Data

- `train/*.avi`, `train.csv` (`id, label`): 277 clip, nhóm g01-g17.
- `test/*.avi`, `test.csv` (`id`): 128 clip, nhóm g18-g25, tên file đã đổi thành id ngẫu nhiên.
- `sample_submission.csv`, `baseline.py` (khung giữa + ResNet-18 + logistic).

Đọc video bằng `cv2.VideoCapture`.

Nguồn: UCF101 (Soomro và cộng sự, 2012), bản 10 lớp `sayakpaul/ucf101-subset`.
