# KTVIC_Captioning - nội dung từng ô trên Kaggle

## Title

OlpAI thử - Sinh caption tiếng Việt cho ảnh

## Competition URL

pka-olpai-test-ktvic-caption

## Subtitle

Viết một câu tiếng Việt mô tả mỗi ảnh đời sống hằng ngày (dạng bài chung kết OlpAI 2025).

## Overview

Sinh một câu mô tả tiếng Việt cho mỗi ảnh, chấm bằng CIDEr-D với 5 câu tham chiếu. Phần ảnh chỉ dùng backbone ImageNet của BTC, phần sinh câu train từ đầu.

## Description

Dạng bài **sinh mô tả ảnh** (đề chung kết OlpAI 2025 có bài này): viết một câu tiếng Việt mô tả nội dung mỗi ảnh. Ảnh là cảnh đời sống hằng ngày ở Việt Nam (đường phố, chợ, quán ăn, nhà cửa...).

Mỗi ảnh train có 5 câu mô tả tiếng Việt do người viết, ví dụ:

```
đây là khung cảnh xuất hiện ở phía trước một căn nhà
có một căn nhà cao tầng xuất hiện ở trong bức ảnh
có một chiếc xe máy xuất hiện ở trong căn nhà
```

**Nhiệm vụ**: với mỗi ảnh test, sinh **một** câu mô tả.

## Evaluation

**CIDEr-D** (cài đặt giống `pycocoevalcap`), so câu của bạn với 5 câu tham chiếu của ảnh:

- Tách từ: chữ thường, bỏ dấu câu, tách theo khoảng trắng (mỗi âm tiết là một từ).
- Mỗi câu thành vector TF-IDF của n-gram 1-4; IDF tính trên các câu tham chiếu của tập đang chấm, nên n-gram xuất hiện ở mọi ảnh ("trong bức ảnh") gần như không được điểm, còn n-gram đặc trưng ("chiếc xe máy", "quầy hàng") được điểm cao.
- Cosine giữa câu của bạn và từng câu tham chiếu, có cắt ngưỡng (lặp một từ nhiều lần không được thêm điểm) và phạt chênh độ dài `exp(-(Δ độ dài)^2 / 72)`; trung bình trên 4 bậc n-gram và 5 tham chiếu, nhân 10.

Thang điểm từ 0 trở lên; nộp một câu tham chiếu đúng của ảnh được khoảng 3,5, nộp cùng một câu chung chung cho mọi ảnh được khoảng 0,17.

## Submission File

`submission.csv` có tiêu đề `id,caption`, mỗi ảnh test một dòng (câu có dấu phẩy thì để trong dấu nháy kép, pandas tự làm):

```
id,caption
te012345,một người đàn ông đang bán hàng ở trên vỉa hè
```

Public là 279 ảnh trong `test/images/`, private là 279 ảnh trong `private_test.zip`. Nên nộp đủ 558 dòng; trước khi có mật khẩu, giữ nguyên các dòng private như `sample_submission.csv`.

## Timeline

- Bắt đầu: do người tổ chức thông báo.
- Giờ 0-5: làm bài, tối đa 20 lần nộp, chỉ có public test.
- Giờ thứ 6: công bố mật khẩu `private_test.zip`; dự đoán private, viết `main.py`, chọn bài nộp cuối.
- Kết thúc: 6 tiếng sau giờ bắt đầu; bảng private được mở ngay sau đó.

## Rules

- Phần ảnh chỉ được dùng weights pretrained ImageNet trong danh sách của BTC, chỉ làm backbone:
  - torchvision (`weights="DEFAULT"`): resnet18, resnet34, resnet50, convnext_tiny, efficientnet_b0, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, regnet_y_400mf, densenet121, shufflenet_v2, squeezenet.
  - timm / Hugging Face: resnet18.a1_in1k, resnet50.a1_in1k, efficientnet_b0, mobilenetv3, microsoft/resnet.
- Không dùng model ngôn ngữ pretrained, không dùng model ảnh-chữ (CLIP, BLIP, SigLIP...), DINOv2, SAM. Phần sinh câu phải train từ đầu.
- Không dùng dữ liệu ngoài bộ được cấp (kể cả bản KTVIC gốc). Không viết caption tay cho ảnh test.
- 5 tiếng đầu: tối đa 20 lần nộp public. Tiếng cuối: mật khẩu `private_test.zip` được công bố, tối đa 5 lần nộp private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Không mở `private_test.zip` trước khi mật khẩu được công bố.
- Tối đa 20 lần nộp; chọn bài nộp cuối để chấm private.

## Citation

Pham và cộng sự, 2024 - KTVIC: A Vietnamese Image Captioning Dataset on the Life Domain. Bản Hugging Face ai-enthusiasm-community/KTVIC.

## Dataset Description

```
train/images/*.jpg     3.769 ảnh
train_captions.json    {id: [5 câu mô tả]}
test/images/*.jpg      279 ảnh public
private_test.zip       (có mật khẩu) private_test/images/*.jpg: 279 ảnh private
sample_submission.csv  id,caption (cùng một câu cho mọi ảnh)
baseline.ipynb, baseline.py
LICENSE-DATASET.txt
```

Giải nén private test (khi có mật khẩu):

```python
import zipfile
zipfile.ZipFile("private_test.zip").extractall(".", pwd=b"<mật khẩu>")   # hoặc: unzip -P <mật khẩu> private_test.zip
```

## Data license

Other (specified in description): KTVIC, chỉ dùng cho học tập và nghiên cứu.

## Files để tải lên

- Data: `KTVIC_Captioning_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py` (hàm `cider_d`), cao hơn là tốt hơn
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Mật khẩu private test: `host/mat_khau_private_test.txt`, công bố ở tiếng cuối
- Kiểm tra bằng sandbox submission: nộp `host/nop_dung_het.csv` (nếu có) phải ra điểm tối đa; nộp `submission.csv` của baseline phải gần `host/min_score.txt` (public 0.1964, private 0.2174)
