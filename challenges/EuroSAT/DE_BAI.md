# Bài luyện tập IMGCLS · Phân loại ảnh vệ tinh khi ít nhãn

*Bài ôn luyện của đội tuyển PHENIKAA AI Hackathon 2026. Dữ liệu: EuroSAT (MIT).*

## 1. Bối cảnh
Một công ty bản đồ cần phân loại lớp phủ đất từ ảnh vệ tinh Sentinel-2 nhưng chỉ có **100 ảnh có nhãn cho mỗi lớp**. Ảnh thực tế lại khác ảnh huấn luyện: bị xoay/lật, nhiễu cảm biến, mờ, lệch màu, nén JPEG mạnh. Hãy xây bộ phân loại bền với những thay đổi đó.

## 2. Nhiệm vụ
Phân loại mỗi ảnh 64×64 RGB vào một trong 10 lớp: `AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake`.

## 3. Dữ liệu (gói `IMGCLS_data.zip`)
| Thư mục / file | Nội dung |
|---|---|
| `train/` + `train_labels.csv` | 1 000 ảnh có nhãn (100/lớp), ảnh sạch |
| `val/` + `val_labels.csv` | 500 ảnh có nhãn (50/lớp), ảnh sạch |
| `unlabeled/` | 3 000 ảnh không nhãn (có thể dùng bán giám sát) |
| `test/` | 4 000 ảnh PNG **đã bị biến đổi** (xoay/lật, nhiễu, mờ, đổi màu, nén JPEG), không nhãn |
| `classes.csv`, `sample_submission.csv`, `starter.py`, `scorer.py` | danh sách lớp, file nộp mẫu, baseline, hàm chấm |

Lưu ý ảnh test **khó hơn** ảnh validation: đừng chỉ tối ưu trên validation sạch; hãy tự tạo biến đổi (augmentation) khi huấn luyện và kiểm tra.

**Giấy phép.** EuroSAT: Helber et al., *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*, IEEE JSTARS 2019. Giấy phép **MIT** (theo bản phát hành của torchgeo trên Hugging Face); ảnh gốc từ Sentinel-2 (Copernicus, miễn phí sử dụng). Gói này là bản lấy mẫu và biến đổi từ dữ liệu gốc.

## 4. File nộp
`submission.csv`, UTF-8, dòng đầu `id,label`, **đúng một dòng cho mỗi ảnh test** (id là tên file không đuôi, ví dụ `t0001`); `label` là tên lớp như trong `classes.csv`.

## 5. Chấm điểm
**Điểm chính: macro-F1** trên các ảnh được chấm; điểm phụ: accuracy. Tập test chia public 40%/private 60% theo ảnh (không biết ảnh nào thuộc phần nào).

## 6. Luật
Được dùng mô hình tiền huấn luyện có giấy phép cho phép (ghi nguồn). Không tìm nhãn test từ bản EuroSAT gốc, không gán nhãn tay cho test. Tối đa 20 lần nộp mỗi ngày.

## 7. Gợi ý
Augmentation mạnh (xoay 90°, nhiễu, blur, jitter màu, JPEG), tinh chỉnh mô hình tiền huấn luyện, học bán giám sát với `unlabeled`, test-time augmentation, kết hợp nhiều mô hình.

## 8. Nguồn cảm hứng
At-Home Task 1 của IOAI 2026 (CC BY 4.0): thích nghi mô hình với dữ liệu mới và ít nhãn mà không làm hỏng khả năng cũ; Kaggle: các cuộc thi phân loại ảnh.
