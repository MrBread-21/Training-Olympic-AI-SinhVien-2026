# cv0 - nội dung từng ô trên Kaggle

## Title

OlpAI thử - CV0 Nordland nhận dạng địa điểm

## Competition URL

pka-olpai-test-nordland

## Subtitle

Tìm ảnh mùa hè chụp cùng vị trí với ảnh mùa đông trên tuyến đường sắt Nordland. Chỉ backbone ImageNet.

## Overview

Mô phỏng lại tác vụ 2 vòng loại OlpAI'26: nhận dạng địa điểm bằng hình ảnh trên bộ Nordland. Với mỗi ảnh chụp mùa đông từ đầu tàu, tìm ảnh mùa hè chụp cùng vị trí. Chỉ dùng backbone ImageNet trong danh sách của BTC.

## Description

### Bối cảnh

Một đoàn tàu chạy cùng một tuyến đường sắt dài 728 km ở Na Uy vào bốn mùa, camera đặt ở đầu tàu. Cùng một vị trí nhưng ảnh mùa đông (tuyết phủ trắng) và ảnh mùa hè (cây xanh) trông rất khác nhau.

### Nhiệm vụ

Với mỗi ảnh truy vấn chụp **mùa đông**, tìm trong cơ sở dữ liệu ảnh **mùa hè** những ảnh chụp cùng vị trí, xếp theo độ khớp giảm dần.

Một truy vấn có thể khớp với nhiều ảnh mùa hè: các khung liền nhau (lệch tối đa 2 khung) được tính là cùng vị trí.

### Ghi chú

Đề gốc chưa nêu rõ cách chia dữ liệu, định dạng nộp và dung sai, nên bản này chọn theo cách làm chuẩn của bài toán: chia theo đoạn hành trình (70% đầu train, 30% cuối test), ảnh test bị xáo trộn thứ tự, chấm Recall@1 với dung sai 2 khung.

## Evaluation

Điểm là **Recall@1**: tỉ lệ truy vấn có ảnh xếp hạng 1 nằm trong tập ảnh đúng (cùng vị trí, lệch tối đa 2 khung).

Baseline (so khớp ảnh thu nhỏ, không học) đạt khoảng **0.07**. Một ResNet-50 ImageNet dùng nguyên, không train thêm, chỉ đạt khoảng 0.03: phải học từ các cặp ảnh đông-hè trong train.

## Submission File

Mỗi truy vấn một dòng. Cột `predictions` gồm tối đa 10 id ảnh database, cách nhau bởi dấu cách, xếp từ khớp nhất. Chỉ ảnh đầu tiên được dùng để tính điểm; các ảnh sau giúp bạn tự tính Recall@5, Recall@10 trên tập val.

```
id,predictions
q000123,d482910 d118273 d903311
q000456,d220019 d220020 d771002
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
- Không dùng model huấn luyện riêng cho nhận dạng địa điểm (NetVLAD, MixVPR...).
- Không dùng dữ liệu ngoài, không gán nhãn tay cho test.
- Tối đa 20 lần nộp; chọn 2 bài nộp cuối để chấm private.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.

## Citation

Sünderhauf, N., Neubert, P., Protzel, P. "Are we there yet? Challenging SeqSLAM on a 3000 km journey across all four seasons." Workshop on Long-Term Autonomy, IEEE ICRA, 2013. Dữ liệu: Hugging Face `Somayeh-h/Nordland`, giấy phép CC BY-NC-SA 4.0.

## Data license

CC BY-NC-SA 4.0

## Dataset Description

Ảnh đã thu về 320x180. Đã bỏ các khung tàu chạy dưới 15 km/h, khung trong đường hầm và 50 khung ở chỗ nối giữa train và test.

### Files

- `train/summer/*.jpg`, `train/winter/*.jpg`: 19.289 cặp ảnh của 70% đầu hành trình. Ảnh cùng tên ở hai mùa là cùng một vị trí.
- `train.csv`: `frame, summer_path, winter_path`. `frame` là thứ tự khung trên hành trình; các khung liền nhau là các vị trí liền nhau.
- `test/query/*.jpg`: 8.253 ảnh truy vấn mùa đông của 30% cuối hành trình, id ngẫu nhiên, đã xáo trộn thứ tự.
- `test/database/*.jpg`: 8.253 ảnh mùa hè của cùng đoạn hành trình, id ngẫu nhiên, đã xáo trộn thứ tự.
- `query.csv`, `database.csv`: danh sách id.
- `sample_submission.csv`: file nộp mẫu (ngẫu nhiên).
- `baseline.py`: baseline so khớp ảnh thu nhỏ. Chạy `python baseline.py /kaggle/input/competitions/pka-olpai-test-nordland`.
