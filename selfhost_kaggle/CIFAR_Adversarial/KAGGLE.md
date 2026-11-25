# CIFAR_Adversarial - nội dung từng ô trên Kaggle

## Title

OlpAI thử - Tấn công đối kháng CIFAR-10

## Competition URL

pka-olpai-test-cifar-adversarial

## Subtitle

Sửa ảnh không quá 8 mức xám để lừa mô hình đã công bố và một mô hình bị giấu (dạng bài chung kết OlpAI 2025).

## Overview

Tạo ảnh đối kháng lệch ảnh gốc không quá 8/255 để mô hình A (công bố) và mô hình B (giấu kín, cùng kiến trúc) đoán sai. Mỗi ảnh hợp lệ được 0,5 cho mỗi mô hình bị lừa.

## Description

Dạng bài **tạo ảnh đối kháng** (đề chung kết OlpAI 2025 có bài này): sửa mỗi ảnh một lượng rất nhỏ, mắt người không thấy khác, để mô hình phân loại đoán sai. Mô hình A được công bố đầy đủ (tấn công "hộp trắng"); một mô hình B cùng kiến trúc được giữ kín và cũng dùng để chấm, nên ảnh đối kháng phải **lừa được cả mô hình chưa thấy** (khả năng chuyển giao).

Ảnh CIFAR-10 cỡ 32 x 32, 10 lớp: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Mô hình A (`model_A.pt`, kiến trúc trong `mo_hinh.py`) là CNN nhỏ train từ đầu, đúng khoảng 81% trên test CIFAR-10. Mô hình B cùng kiến trúc, train trên cùng dữ liệu với seed khác, không công bố.

1.000 ảnh test (100 ảnh mỗi lớp) đều được cả A và B phân loại đúng.

**Nhiệm vụ**: với mỗi ảnh test, nộp một ảnh mới:

- mỗi giá trị điểm ảnh (mỗi kênh màu) lệch ảnh gốc **không quá 8** trên thang 0-255;
- làm A và B đoán sai nhãn thật (tấn công không chỉ định lớp đích).

## Evaluation

Mỗi ảnh:

- **Không hợp lệ** (không giải mã được thành đúng 3.072 byte, hoặc có điểm ảnh lệch ảnh gốc quá 8): 0 điểm.
- **Hợp lệ**: 0,5 điểm nếu mô hình A đoán sai nhãn thật, cộng 0,5 điểm nếu mô hình B đoán sai nhãn thật.

Điểm cuối là trung bình trên các ảnh, thang 0-1. Nộp lại ảnh gốc được 0. Metric chạy mô hình bằng numpy với đúng trọng số trong `model_A.pt`, nên kết quả của A trên máy bạn trùng với lúc chấm.

## Submission File

`submission.csv` có tiêu đề `id,image`. `image` là chuỗi base64 của 3.072 byte `uint8`: ảnh 32 x 32 x 3 theo thứ tự hàng, cột, kênh (HWC, RGB), đúng như `np.asarray(Image.open(...))`:

```python
import base64, numpy as np, pandas as pd
# adv: mảng uint8 (N, 32, 32, 3)
pd.DataFrame({"id": ids, "image": [base64.b64encode(a.tobytes()).decode() for a in adv]}).to_csv("submission.csv", index=False)
```

Public là 500 ảnh trong `test/images/`, private là 500 ảnh trong `private_test.zip`. Nên nộp đủ 1.000 dòng; trước khi có mật khẩu, giữ nguyên các dòng private như `sample_submission.csv` (ảnh gốc, 0 điểm).

## Timeline

- Bắt đầu: do người tổ chức thông báo.
- Giờ 0-5: làm bài, tối đa 20 lần nộp, chỉ có public test.
- Giờ thứ 6: công bố mật khẩu `private_test.zip`; dự đoán private, viết `main.py`, chọn bài nộp cuối.
- Kết thúc: 6 tiếng sau giờ bắt đầu; bảng private được mở ngay sau đó.

## Rules

- Được dùng `model_A.pt` tuỳ ý. Được tự train thêm mô hình thay thế (surrogate) trên `cifar_train.npz`, kể cả dùng weights ImageNet trong danh sách của BTC.
- Không dùng dữ liệu hay mô hình ngoài bộ được cấp. Không dò mô hình B bằng cách nộp bài liên tục ngoài giới hạn lượt nộp.
- 5 tiếng đầu: tối đa 20 lần nộp public. Tiếng cuối: mật khẩu `private_test.zip` được công bố, tối đa 5 lần nộp private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Không mở `private_test.zip` trước khi mật khẩu được công bố.
- Tối đa 20 lần nộp; chọn bài nộp cuối để chấm private.

## Citation

Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. Bản Hugging Face uoft-cs/cifar10.

## Dataset Description

```
model_A.pt             trọng số mô hình A (state_dict PyTorch)
mo_hinh.py             kiến trúc TinyCNN và hàm tai_mo_hinh(); đầu vào tensor (N, 3, 32, 32) giá trị 0-1
test/images/*.png      500 ảnh public
test_labels.csv        id,label của 500 ảnh public (label là số 0-9 theo thứ tự lớp ở trên)
private_test.zip       (có mật khẩu) private_test/images/*.png và private_test/test_labels.csv: 500 ảnh private
cifar_train.npz        50.000 ảnh train CIFAR-10: x (uint8, 50000 x 32 x 32 x 3), y (nhãn)
sample_submission.csv  id,image (ảnh gốc)
baseline.ipynb, baseline.py
LICENSE-DATASET.txt
```

Giải nén private test (khi có mật khẩu):

```python
import zipfile
zipfile.ZipFile("private_test.zip").extractall(".", pwd=b"<mật khẩu>")   # hoặc: unzip -P <mật khẩu> private_test.zip
```

## Data license

Other (specified in description): CIFAR-10, dùng cho học tập và nghiên cứu.

## Files để tải lên

- Data: `CIFAR_Adversarial_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py` (hàm `adv_success`), cao hơn là tốt hơn
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Mật khẩu private test: `host/mat_khau_private_test.txt`, công bố ở tiếng cuối
- Kiểm tra bằng sandbox submission: nộp `host/nop_dung_het.csv` (nếu có) phải ra điểm tối đa; nộp `submission.csv` của baseline phải gần `host/min_score.txt` (public 0.8090, private 0.8080)
