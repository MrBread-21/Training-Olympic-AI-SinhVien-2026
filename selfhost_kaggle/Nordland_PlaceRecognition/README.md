# Định vị tàu hoả bốn mùa (Nordland Visual Place Recognition)

Dựng lại **tác vụ 2 vòng sơ loại OlpAI'26** theo đúng văn bản đề chính thức: truy vấn mùa hè, kho ảnh mùa đông, dung sai ±10 khung, file nộp `query_id,winter_id`, Recall@1 thang 0-100, private test trong zip có mật khẩu. Dữ liệu được cắt lại từ bản Nordland công khai nên các đoạn tuyến và ảnh cụ thể khác với đề thật, còn quy mô, định dạng và cách chấm giữ như đề.

## Overview

Định vị bằng hình ảnh (Visual Place Recognition - VPR) là bài toán xác định vị trí của thiết bị chỉ dựa trên ảnh camera, bằng cách đối chiếu ảnh thu được với một kho ảnh đã biết vị trí. Đây là thành phần quan trọng của đầu tàu tự hành, robot kiểm tra đường ray hay thiết bị bay không người lái khi tín hiệu định vị vệ tinh bị mất trong núi hoặc đường hầm.

Dữ liệu là tuyến đường sắt Nordland (Na Uy, khoảng 729 km) do đài NRK quay bằng camera gắn ở đầu tàu, một lần vào mùa hè và một lần vào mùa đông. **Kho ảnh tham chiếu được lập ở mùa đông, còn ảnh cần định vị được chụp ở mùa hè**, khi cảnh vật đã thay đổi gần như hoàn toàn. Ảnh cần định vị có thể bị tối, nhoè, sương mù, nén mạnh, độ phân giải thấp, gặp mưa hay chói sáng.

**Nhiệm vụ**: với mỗi ảnh mùa hè trong `queries/`, chọn **đúng một** ảnh mùa đông trong `database/` được chụp ở cùng vị trí trên tuyến. Nhiều ảnh mùa hè được phép có cùng một dự đoán.

**Luật (giống vòng sơ loại và khu vực OlpAI'26)**:

- Chỉ dùng weights pretrained ImageNet trong danh sách sau, không tải model khác:
  - torchvision (`weights="DEFAULT"`): resnet18, resnet34, resnet50, convnext_tiny, efficientnet_b0, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, regnet_y_400mf, densenet121, shufflenet_v2, squeezenet.
  - timm / Hugging Face: resnet18.a1_in1k, resnet50.a1_in1k, efficientnet_b0, mobilenetv3, microsoft/resnet.
- Không dùng model nền tảng ảnh-chữ (CLIP, SigLIP), DINOv2, VideoMAE, SAM, hay model huấn luyện riêng cho VPR (NetVLAD, MixVPR, BoQ có trọng số sẵn...). Được tự xây kiến trúc (kể cả BoQ, NetVLAD) và train từ đầu trên dữ liệu đề.
- Không dùng ảnh hay dữ liệu ngoài bộ được cấp, kể cả bản quay Nordland gốc và các mùa khác.
- 5 tiếng đầu: tối đa 20 lần nộp public. Tiếng cuối: mật khẩu `private_test.zip` được công bố, tối đa 5 lần nộp private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.

## Evaluation

**Recall@1** (thang 0-100): tỉ lệ ảnh mùa hè được định vị đúng.

```
Recall@1 = 100/n x số ảnh q có |pos(ŵ) - pos(q)| ≤ 10
```

- `pos` là nhãn `position`: số thứ tự khung hình của bản quay mùa đông. Ảnh mùa hè q và ảnh mùa đông w được coi là cùng một chỗ nếu lệch không quá 10 khung, nên mỗi ảnh mùa hè có nhiều ảnh mùa đông được tính là đúng.
- n là số ảnh mùa hè của tập đang chấm, không phụ thuộc số dòng trong file.

File `output.csv` có tiêu đề `query_id,winter_id` (tên file ảnh, bỏ đuôi `.jpg`), mỗi ảnh mùa hè một dòng:

```
query_id,winter_id
q_7d1a0c,w_03be91
```

Khi chấm:

- `query_id` xuất hiện nhiều dòng thì chỉ chấm dòng đầu tiên.
- Dòng có `query_id` không thuộc tập kiểm tra bị bỏ qua.
- Ảnh mùa hè không có dòng nào, hoặc có `winter_id` không thuộc kho, tính là định vị sai.
- File thiếu cột hoặc sai tên cột được 0 điểm.

Ví dụ: 4 ảnh mùa hè có vị trí thật 9053, 9123, 9300, 9412; ảnh mùa đông dự đoán có vị trí 9045, 9140, 9300, 9405. Sai lệch 8, 17, 0, 7, nên Recall@1 = 100 x 3/4 = 75,00.

Trên Kaggle: phần public là 450 ảnh trong `test/queries/`, phần private là 460 ảnh trong `private_test.zip`. File nộp nên có đủ 910 dòng; trước khi có mật khẩu, các dòng private để nguyên như `sample_submission.csv`.

## Data

Mọi ảnh là `.jpg` 640x360. Tên file là chuỗi ngẫu nhiên; tên và thứ tự ảnh không mang thông tin vị trí. Tuyến được chia thành 4 đoạn không giao nhau (cách nhau 400 khung) cho train, validation, public test và private test. Mỗi đoạn có 1 ảnh mùa hè cho mỗi 10 khung, còn kho mùa đông giữ mọi khung liên tiếp (khoảng 10 ảnh đông cho mỗi ảnh hè, nhiều ảnh gần như giống nhau).

```
train/summer/*.jpg            1.400 ảnh mùa hè, giữ nguyên chất lượng
train/winter/*.jpg            14.000 ảnh mùa đông
train_summer.csv              summer_id,position
train_winter.csv              winter_id,position
validation/summer/*.jpg       300 ảnh mùa hè (bị làm hỏng như test)
validation/winter/*.jpg       3.000 ảnh mùa đông, đoạn tuyến khác
validation_summer.csv, validation_winter.csv
test/queries/*.jpg            450 ảnh mùa hè của public test (bị làm hỏng)
test/database/*.jpg           9.100 ảnh mùa đông của cả public và private test, trộn chung
private_test.zip              (có mật khẩu) private_test/queries/*.jpg: 460 ảnh mùa hè (bị làm hỏng)
sample_submission.csv         query_id,winter_id cho cả 910 ảnh (dự đoán ngẫu nhiên)
baseline.ipynb, baseline.py   baseline so khớp ảnh thu nhỏ, có tự chấm trên validation
LICENSE-DATASET.txt
```

Giải nén private test (khi có mật khẩu):

```python
import zipfile
zipfile.ZipFile("private_test.zip").extractall(".", pwd=b"<mật khẩu>")   # hoặc: unzip -P <mật khẩu> private_test.zip
```

## Khác biệt với đề thật

| | Đề thật | Bản này |
|---|---|---|
| Đoạn tuyến, ảnh cụ thể | Do BTC chọn | Cắt từ 27.592 khung đã lọc (bỏ khung tàu dừng, chạy chậm, trong hầm) của bản Hugging Face |
| Kho mùa đông của test | 9.098 ảnh | 9.100 ảnh |
| Cách làm hỏng ảnh hè | Không công bố | Mỗi ảnh 1-2 kiểu trong 7 kiểu đề nêu (tối, nhoè, sương mù, nén JPEG mạnh, độ phân giải thấp, mưa, chói), áp cho validation và test |
| Nộp private | Nộp riêng 5 lần ở tiếng cuối | Kaggle chấm cả 910 dòng trong một file, cột `Usage` tách public/private |

Nguồn: Nordland (Sünderhauf, Neubert, Protzel, ICRA 2013 workshop), bản Hugging Face `Somayeh-h/Nordland`, giấy phép CC BY-NC-SA 4.0. Đề gốc: OlpAI'26 vòng sơ loại, tác vụ 2.
