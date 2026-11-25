# cv5_da_nhan - nội dung từng ô trên Kaggle

## Title

OlpAI thử - CV5 Phân loại đa nhãn

## Competition URL

olpai-thu5-da-nhan

## Subtitle

Dự đoán ảnh có những vật nào trong 20 lớp, chọn ngưỡng cho từng lớp. Chỉ backbone ImageNet.

## Overview

Dự đoán tập vật thể có trong ảnh (20 lớp), chấm bằng macro-F1 đa nhãn. Chỉ dùng backbone ImageNet trong danh sách của BTC.

## Description

Mỗi ảnh có thể chứa một hoặc nhiều vật thuộc 20 lớp (người, chó, ô tô, ghế, chai...). Dự đoán tập lớp có trong ảnh.

Khó khăn chính là **chọn ngưỡng**: mô hình cho xác suất từng lớp, nhưng metric chấm theo nhãn cứng, và các lớp hiếm (chai, chậu cây, ghế) cần ngưỡng khác lớp phổ biến (người).

## Evaluation

**Macro-F1 đa nhãn**: với mỗi lớp tính F1 trên toàn bộ ảnh test (lớp đó có hay không có trong ảnh), rồi lấy trung bình 20 lớp.

Cột `labels`: tên lớp cách nhau bởi dấu cách, đúng chính tả như `classes.csv`. Được để trống nếu không có lớp nào.

```
id,labels
te123456,person dog
te654321,car
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
- Không gán nhãn tay cho test.
- Tối đa 20 lần nộp; chọn 2 bài nộp cuối để chấm private.

## Citation

PASCAL VOC 2012 (Everingham và cộng sự), phần có mask phân đoạn; nhãn lấy từ các lớp xuất hiện trong mask.

## Dataset Description

- `train/*.jpg`, `train.csv` (`id, labels`): 1.464 ảnh.
- `test/*.jpg`, `test.csv` (`id`): 1.449 ảnh.
- `classes.csv`: 20 lớp.
- `sample_submission.csv`, `baseline.py` (chạy `python baseline.py /kaggle/input/olpai-thu5-da-nhan`) (ResNet-18 đóng băng + 20 logistic, ngưỡng 0.5).

Ảnh đã thu về cạnh dài 384 px.

## Data license

Other: PASCAL VOC 2012, ảnh Flickr theo điều khoản của Flickr, dùng cho nghiên cứu

## Files để tải lên

- Data: `cv5_da_nhan_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py`
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Kiểm tra bằng sandbox submission: nộp file từ `baseline.py`, điểm phải gần `host/min_score.txt`
