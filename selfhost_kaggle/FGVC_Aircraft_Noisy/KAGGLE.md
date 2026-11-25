# cv1_may_bay - nội dung từng ô trên Kaggle

## Title

OlpAI thử - CV1 Phân loại họ máy bay

## Competition URL

olpai-thu5-may-bay

## Subtitle

Phân loại 70 họ máy bay khi 15% nhãn train sai và ảnh test bị làm hỏng. Chỉ backbone ImageNet.

## Overview

Phân loại ảnh máy bay vào 70 họ khi nhãn train có nhiễu và hơn nửa ảnh test bị nén, mờ, nhiễu hoặc che. Chỉ dùng backbone ImageNet trong danh sách của BTC.

## Description

Phân loại ảnh máy bay vào 70 họ (Boeing 737, A320, ATR-72, Cessna 172...). Đây là bài phân loại chi tiết: các lớp rất giống nhau, chỉ khác ở số cửa sổ, hình đuôi, vị trí động cơ.

Hai khó khăn cố ý:

- **Khoảng 15% nhãn train bị sai**, thường bị gán nhầm sang họ khác cùng hãng (ví dụ A330 thành A340).
- **Khoảng 60% ảnh test bị làm hỏng**: nén JPEG nặng, mờ, nhiễu, đổi sáng tối, bị che một mảng. Ảnh train thì sạch.

## Evaluation

**Macro-F1** trên 70 lớp.

```
id,label
te123456,Boeing 737
te654321,A320
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

FGVC-Aircraft (Maji và cộng sự, 2013), chỉ dùng cho nghiên cứu phi thương mại.

## Dataset Description

- `train/*.jpg` và `train.csv` (`id, label`): 3.334 ảnh, nhãn có nhiễu.
- `test/*.jpg` và `test.csv` (`id`): 3.333 ảnh.
- `classes.csv`: tên 70 họ, `label` trong file nộp phải đúng chính tả như ở đây.
- `sample_submission.csv`, `baseline.py` (chạy `python baseline.py /kaggle/input/olpai-thu5-may-bay`) (đặc trưng ResNet-18 đóng băng + logistic).

Ảnh đã bỏ dải chữ bản quyền ở đáy và thu về cạnh dài 400 px.

## Data license

Other: FGVC-Aircraft, chỉ dùng cho nghiên cứu phi thương mại

## Files để tải lên

- Data: `cv1_may_bay_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py`
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Kiểm tra bằng sandbox submission: nộp file từ `baseline.py`, điểm phải gần `host/min_score.txt`
