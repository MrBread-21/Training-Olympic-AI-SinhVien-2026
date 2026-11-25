# cv2_polyp - nội dung từng ô trên Kaggle

## Title

OlpAI thử - CV2 Phân đoạn polyp

## Competition URL

olpai-thu5-polyp

## Subtitle

Khoanh vùng polyp trong ảnh nội soi đại tràng, nộp mask dạng RLE. Chỉ backbone ImageNet.

## Overview

Phân đoạn polyp ở mức điểm ảnh trong ảnh nội soi đại tràng, chấm bằng Dice. Chỉ dùng backbone ImageNet trong danh sách của BTC.

## Description

Cho ảnh nội soi đại tràng, khoanh vùng polyp (khối u nhỏ) ở mức từng điểm ảnh. Polyp có kích thước, hình dạng, màu rất khác nhau và nhiều khi lẫn với nếp niêm mạc.

## Evaluation

**Dice trung bình theo ảnh**: Dice = 2|A ∩ B| / (|A| + |B|); ảnh mà cả dự đoán và đáp án đều rỗng tính Dice = 1.

Mask nộp dạng **RLE**: chuỗi `start length start length ...`, đánh số điểm ảnh từ 1, **duyệt theo hàng** (trái sang phải, rồi xuống hàng dưới), trên ảnh 256x256. Mask rỗng thì để trống.

```python
def rle_encode(mask):  # mask: mảng (256, 256) gồm 0/1
    p = np.concatenate([[0], mask.flatten(), [0]])
    r = np.where(p[1:] != p[:-1])[0] + 1
    r[1::2] -= r[::2]
    return " ".join(map(str, r))
```

```
id,rle
im123456,1234 10 1490 12 1746 15
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

Kvasir-SEG (Jha và cộng sự, MMM 2020), dùng cho nghiên cứu và giáo dục.

## Dataset Description

- `train/images/*.jpg`, `train/masks/*.png` (255 là polyp), `train.csv`: 800 ảnh 256x256.
- `test/images/*.jpg`, `test.csv`: 200 ảnh.
- `sample_submission.csv` (một hình vuông ở giữa), `baseline.py` (chạy `python baseline.py /kaggle/input/olpai-thu5-polyp`) (U-Net nhỏ train từ đầu 8 epoch).

## Data license

Other: Kvasir-SEG, dùng cho nghiên cứu và giáo dục

## Files để tải lên

- Data: `cv2_polyp_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py`
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Kiểm tra bằng sandbox submission: nộp file từ `baseline.py`, điểm phải gần `host/min_score.txt`
