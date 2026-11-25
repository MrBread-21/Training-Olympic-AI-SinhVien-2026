# Nordland_PlaceRecognition - nội dung từng ô trên Kaggle

## Title

OlpAI thử - Định vị tàu hoả bốn mùa

## Competition URL

pka-olpai-test-nordland

## Subtitle

Với mỗi ảnh mùa hè, tìm ảnh mùa đông cùng vị trí trên tuyến đường sắt Nordland (tác vụ 2 vòng sơ loại OlpAI'26).

## Overview

Dựng lại tác vụ 2 vòng sơ loại OlpAI'26 "Định vị tàu hoả bốn mùa": với mỗi ảnh mùa hè (có thể bị tối, nhoè, sương mù, nén mạnh...), chọn đúng một ảnh mùa đông trong kho được chụp ở cùng vị trí. Đúng nếu lệch tối đa 10 khung. Chỉ dùng backbone ImageNet trong danh sách của BTC.

## Description

### Bối cảnh

Định vị bằng hình ảnh (Visual Place Recognition) xác định vị trí của thiết bị chỉ dựa trên ảnh camera, bằng cách đối chiếu với một kho ảnh đã biết vị trí. Đây là thành phần quan trọng của đầu tàu tự hành hay robot kiểm tra đường ray khi tín hiệu định vị vệ tinh bị mất trong núi hoặc đường hầm.

Dữ liệu là tuyến đường sắt Nordland (Na Uy, khoảng 729 km) do đài NRK quay bằng camera ở đầu tàu vào mùa hè và mùa đông. **Kho ảnh tham chiếu lập ở mùa đông, ảnh cần định vị chụp ở mùa hè**, khi cảnh vật đã thay đổi gần như hoàn toàn. Ảnh cần định vị có thể bị tối, nhoè, sương mù, nén mạnh, độ phân giải thấp, gặp mưa hay chói sáng.

### Nhiệm vụ

Với mỗi ảnh mùa hè trong `queries/`, chọn **đúng một** ảnh mùa đông trong `database/` được chụp ở cùng vị trí trên tuyến. Nhiều ảnh mùa hè được phép có cùng một dự đoán.

### Khác biệt với đề thật

Dữ liệu được cắt lại từ bản Nordland công khai (27.592 khung đã bỏ đoạn tàu dừng và đường hầm) nên đoạn tuyến và ảnh cụ thể khác đề thật; quy mô, định dạng, cách chấm giữ như đề. Kiểu làm hỏng ảnh mùa hè do người dựng đề tự chọn theo 7 kiểu đề nêu, áp cho validation và test.

## Evaluation

**Recall@1** (thang 0-100): tỉ lệ ảnh mùa hè được định vị đúng. Ảnh mùa hè q và ảnh mùa đông w được coi là cùng một chỗ nếu `|pos(w) - pos(q)| ≤ 10`, với `pos` là số thứ tự khung hình của bản quay mùa đông; mỗi ảnh mùa hè vì vậy có nhiều ảnh mùa đông được tính là đúng.

Khi chấm:

- `query_id` xuất hiện nhiều dòng thì chỉ chấm dòng đầu tiên.
- Dòng có `query_id` không thuộc tập kiểm tra bị bỏ qua.
- Ảnh mùa hè không có dòng nào, hoặc có `winter_id` không thuộc kho, tính là định vị sai.
- File thiếu cột hoặc sai tên cột được 0 điểm.

Ví dụ: 4 ảnh mùa hè có vị trí thật 9053, 9123, 9300, 9412; ảnh mùa đông dự đoán có vị trí 9045, 9140, 9300, 9405. Sai lệch 8, 17, 0, 7, nên Recall@1 = 100 x 3/4 = 75,00.

## Submission File

File `output.csv` có tiêu đề `query_id,winter_id`, mỗi ảnh mùa hè một dòng (tên file ảnh, bỏ đuôi `.jpg`):

```
query_id,winter_id
q_7d1a0c,w_03be91
```

Public là 450 ảnh trong `test/queries/`, private là 460 ảnh trong `private_test.zip`. Nên nộp đủ 910 dòng; trước khi có mật khẩu, giữ nguyên các dòng private như trong `sample_submission.csv`.

## Timeline

- Bắt đầu: thứ 5, 26/11/2026 (giờ bắt đầu do người tổ chức thông báo).
- Giờ 0-5: làm bài, tối đa 20 lần nộp, chỉ có public test.
- Giờ thứ 6: công bố mật khẩu `private_test.zip`; dự đoán private, viết `main.py`, chọn bài nộp cuối.
- Kết thúc: 6 tiếng sau giờ bắt đầu; bảng private được mở ngay sau đó.

## Rules

- Chỉ dùng weights pretrained ImageNet trong danh sách sau, không tải model khác:
  - torchvision (`weights="DEFAULT"`): resnet18, resnet34, resnet50, convnext_tiny, efficientnet_b0, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, regnet_y_400mf, densenet121, shufflenet_v2, squeezenet.
  - timm / Hugging Face: resnet18.a1_in1k, resnet50.a1_in1k, efficientnet_b0, mobilenetv3, microsoft/resnet.
- Không dùng model nền tảng ảnh-chữ (CLIP, SigLIP), DINOv2, VideoMAE, SAM, hay model huấn luyện riêng cho VPR (NetVLAD, MixVPR, BoQ có trọng số sẵn...). Được tự xây kiến trúc và train từ đầu trên dữ liệu đề.
- Không dùng ảnh hay dữ liệu ngoài bộ được cấp, kể cả bản quay Nordland gốc và các mùa khác. Không gán nhãn tay cho test.
- Không mở `private_test.zip` trước khi mật khẩu được công bố.
- Tối đa 20 lần nộp; chọn bài nộp cuối để chấm private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.

## Citation

Sünderhauf, N., Neubert, P., Protzel, P. "Are we there yet? Challenging SeqSLAM on a 3000 km journey across all four seasons." Workshop on Long-Term Autonomy, IEEE ICRA, 2013. Dữ liệu: Hugging Face Somayeh-h/Nordland, giấy phép CC BY-NC-SA 4.0. Đề gốc: Olympic Trí tuệ nhân tạo sinh viên Việt Nam 2026, vòng sơ loại, tác vụ 2.

## Dataset Description

Mọi ảnh là `.jpg` 640x360. Tên file là chuỗi ngẫu nhiên; tên và thứ tự ảnh không mang thông tin vị trí. Tuyến chia thành 4 đoạn không giao nhau cho train, validation, public test, private test. Mỗi đoạn có 1 ảnh mùa hè cho mỗi 10 khung, còn kho mùa đông giữ mọi khung liên tiếp (khoảng 10 ảnh đông cho mỗi ảnh hè).

### Files

- `train/summer/*.jpg` (1.400, giữ nguyên chất lượng), `train/winter/*.jpg` (14.000); nhãn trong `train_summer.csv` (`summer_id,position`) và `train_winter.csv` (`winter_id,position`).
- `validation/summer/*.jpg` (300, bị làm hỏng như test), `validation/winter/*.jpg` (3.000) của một đoạn tuyến khác; nhãn trong `validation_summer.csv`, `validation_winter.csv`.
- `test/queries/*.jpg`: 450 ảnh mùa hè của public test.
- `test/database/*.jpg`: 9.100 ảnh mùa đông của cả public và private test, trộn chung.
- `private_test.zip` (có mật khẩu): `private_test/queries/*.jpg`, 460 ảnh mùa hè. Giải nén: `zipfile.ZipFile("private_test.zip").extractall(".", pwd=b"<mật khẩu>")`.
- `sample_submission.csv`: `query_id,winter_id` cho cả 910 ảnh.
- `baseline.ipynb` / `baseline.py`: so khớp ảnh thu nhỏ, có tự chấm Recall@1 trên validation; trên Kaggle tự tìm dữ liệu trong `/kaggle/input`.

## Data license

CC BY-NC-SA 4.0

## Files để tải lên

- Data: `Nordland_PlaceRecognition_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py` (hàm `recall_at_1_vpr`)
- Solution: `host/solution.csv` (cột id `query_id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Mật khẩu private test: `host/mat_khau_private_test.txt`, công bố ở tiếng cuối
- Kiểm tra bằng sandbox submission: nộp `host/nop_dung_het.csv` phải ra 100, nộp `output.csv` của baseline phải gần `host/min_score.txt`
