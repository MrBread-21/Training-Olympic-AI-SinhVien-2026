# CV0 - Nhận dạng địa điểm qua bốn mùa (Nordland)

Mô phỏng lại tác vụ 2 vòng loại OlpAI'26. Những chỗ đề gốc chưa rõ (cách chia dữ liệu, định dạng nộp, metric, dung sai) được chọn theo cách làm chuẩn của bài toán nhận dạng địa điểm bằng hình ảnh (Visual Place Recognition), xem mục "Giả định" ở cuối.

## Overview

Một đoàn tàu chạy cùng một tuyến đường sắt dài 728 km ở Na Uy vào bốn mùa, camera đặt ở đầu tàu. Cùng một vị trí nhưng ảnh mùa đông (tuyết phủ trắng) và ảnh mùa hè (cây xanh) trông rất khác nhau.

**Nhiệm vụ**: với mỗi ảnh truy vấn chụp mùa đông, tìm trong cơ sở dữ liệu ảnh mùa hè những ảnh chụp cùng vị trí, xếp theo độ khớp giảm dần.

Một truy vấn mùa đông có thể khớp với nhiều ảnh mùa hè: các khung liền nhau (lệch tối đa 2 khung) được tính là cùng vị trí.

**Luật (giống vòng khu vực OlpAI'26)**:

- Chỉ dùng weights pretrained ImageNet trong danh sách sau, không tải model khác:
  - torchvision (`weights="DEFAULT"`): resnet18, resnet34, resnet50, convnext_tiny, efficientnet_b0, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, regnet_y_400mf, densenet121, shufflenet_v2, squeezenet.
  - timm / Hugging Face: resnet18.a1_in1k, resnet50.a1_in1k, efficientnet_b0, mobilenetv3, microsoft/resnet.
- Không dùng model nền tảng ảnh-chữ (CLIP, SigLIP), DINOv2, VideoMAE, SAM.
- Không dùng dữ liệu ngoài.
- Cuối giờ phải có `main.py` chạy lại ra file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Không dùng model huấn luyện riêng cho nhận dạng địa điểm (NetVLAD, MixVPR...).

## Evaluation

**Recall@1**: tỉ lệ truy vấn có ảnh xếp hạng 1 nằm trong tập ảnh đúng (cùng vị trí, lệch tối đa 2 khung).

File nộp có cột `id` (id truy vấn) và `predictions`: tối đa 10 id ảnh database, cách nhau bởi dấu cách, xếp từ khớp nhất. Chỉ ảnh đầu tiên được dùng để tính điểm, các ảnh sau giúp bạn tự đánh giá Recall@5, Recall@10 trên tập val tự chia.

```
id,predictions
q000123,d482910 d118273 d903311
q000456,d220019 d220020 d771002
```

## Data

- `train/summer/*.jpg`, `train/winter/*.jpg`: 19.289 cặp ảnh của 70% đầu hành trình. Ảnh cùng tên ở hai mùa là cùng một vị trí.
- `train.csv`: `frame, summer_path, winter_path`. `frame` là thứ tự khung trên hành trình, các khung liền nhau là các vị trí liền nhau.
- `test/query/*.jpg`: 8.253 ảnh truy vấn mùa đông của 30% cuối hành trình, id ngẫu nhiên, đã xáo trộn thứ tự.
- `test/database/*.jpg`: 8.253 ảnh mùa hè của cùng đoạn hành trình, id ngẫu nhiên, đã xáo trộn thứ tự.
- `query.csv`, `database.csv`: danh sách id.
- `sample_submission.csv`: file nộp mẫu (ngẫu nhiên).
- `baseline.py`: so khớp ảnh thu nhỏ, không học.

Ảnh đã thu về 320x180. Đã bỏ các khung tàu chạy dưới 15 km/h, khung trong đường hầm và 50 khung ở chỗ nối giữa train và test.

## Giả định (sửa khi biết đề thật)

| Chỗ đề gốc chưa rõ | Đang chọn |
|---|---|
| Chia train/test | Theo đoạn hành trình: 70% đầu train (có cả đông và hè), 30% cuối test |
| Thứ tự ảnh test | Xáo trộn, id ngẫu nhiên (không dùng được so khớp chuỗi) |
| Định dạng nộp | `id, predictions` (tối đa 10 id) |
| Metric và dung sai | Recall@1, đúng nếu lệch tối đa 2 khung |

Nguồn: Nordland (Sünderhauf và cộng sự, ICRA 2013 workshop), bản trên HuggingFace `Somayeh-h/Nordland`, giấy phép CC BY-NC-SA 4.0.
