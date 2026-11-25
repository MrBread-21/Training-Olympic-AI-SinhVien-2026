# CV2 - Phân đoạn polyp trong ảnh nội soi

## Overview

Cho ảnh nội soi đại tràng, khoanh vùng polyp (khối u nhỏ) ở mức từng điểm ảnh. Polyp có kích thước, hình dạng, màu rất khác nhau và nhiều khi lẫn với nếp niêm mạc.

**Luật (giống vòng khu vực OlpAI'26)**:

- Chỉ dùng weights pretrained ImageNet trong danh sách sau, không tải model khác:
  - torchvision (`weights="DEFAULT"`): resnet18, resnet34, resnet50, convnext_tiny, efficientnet_b0, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, regnet_y_400mf, densenet121, shufflenet_v2, squeezenet.
  - timm / Hugging Face: resnet18.a1_in1k, resnet50.a1_in1k, efficientnet_b0, mobilenetv3, microsoft/resnet.
- Không dùng model nền tảng ảnh-chữ (CLIP, SigLIP), DINOv2, VideoMAE, SAM.
- Không dùng dữ liệu ngoài.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.

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

## Data

- `train/images/*.jpg`, `train/masks/*.png` (255 là polyp), `train.csv`: 800 ảnh 256x256.
- `test/images/*.jpg`, `test.csv`: 200 ảnh.
- `sample_submission.csv` (một hình vuông ở giữa), `baseline.py` (U-Net nhỏ train từ đầu 8 epoch).

Nguồn: Kvasir-SEG (Jha và cộng sự, MMM 2020), dùng cho nghiên cứu và giáo dục.
