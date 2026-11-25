# cv4_video - nội dung từng ô trên Kaggle

## Title

OlpAI thử - CV4 Nhận dạng hành động video

## Competition URL

olpai-thu5-video

## Subtitle

Phân loại clip video vào 10 hành động, test là các video gốc chưa gặp. Chỉ backbone ImageNet.

## Overview

Phân loại clip video ngắn vào 10 hành động với dữ liệu ít; test gồm các video gốc không có trong train. Chỉ dùng backbone ImageNet trong danh sách của BTC.

## Description

Phân loại clip video ngắn vào 10 hành động (ném bóng chày, úp rổ, chơi bi-a, em bé bò...). Dữ liệu ít: chưa tới 300 clip train.

**Bẫy cần chú ý**: trong train, mỗi nhóm `gXX` (tên file `v_<Lớp>_gXX_cYY.avi`) là các clip cắt từ **cùng một video gốc**, nên rất giống nhau. Test gồm các nhóm hoàn toàn khác train. Chia val ngẫu nhiên theo clip sẽ cho điểm val ảo cao; phải chia theo nhóm.

## Evaluation

**Macro-F1** trên 10 lớp.

```
id,label
v123456,BasketballDunk
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
- Không dùng model video huấn luyện trên Kinetics (r3d_18, x3d...): chỉ 2D CNN ImageNet trên từng khung rồi gộp theo thời gian.
- Không gán nhãn tay cho test.
- Tối đa 20 lần nộp; chọn 2 bài nộp cuối để chấm private.

## Citation

UCF101 (Soomro và cộng sự, 2012), bản 10 lớp `sayakpaul/ucf101-subset`.

## Dataset Description

- `train/*.avi`, `train.csv` (`id, label`): 277 clip, nhóm g01-g17.
- `test/*.avi`, `test.csv` (`id`): 128 clip, nhóm g18-g25, tên file đã đổi thành id ngẫu nhiên.
- `sample_submission.csv`, `baseline.py` (chạy `python baseline.py /kaggle/input/olpai-thu5-video`) (khung giữa + ResNet-18 + logistic).

Đọc video bằng `cv2.VideoCapture`.

## Data license

Other: UCF101, dùng cho nghiên cứu (bản con sayakpaul/ucf101-subset, Apache 2.0)

## Files để tải lên

- Data: `cv4_video_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py`
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Kiểm tra bằng sandbox submission: nộp file từ `baseline.py`, điểm phải gần `host/min_score.txt`
