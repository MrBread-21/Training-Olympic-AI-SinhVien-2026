# Bài luyện tập CCTV · Phát hiện bất thường trên camera giám sát

*Bài ôn luyện của đội tuyển PHENIKAA AI Hackathon 2026. Dữ liệu: MERL Street Scene (CC BY-SA 4.0).*

## 1. Bối cảnh
Một camera USB cố định nhìn xuống một đoạn phố hai làn xe, có làn xe đạp và vỉa hè. Phần lớn thời gian là sinh hoạt bình thường (xe chạy, người đi bộ, người đẩy xe nôi, người chạy bộ, bóng đổ thay đổi, cờ và cây lay động). Thỉnh thoảng có **sự kiện bất thường**: đi sai làn, người băng qua đường ngoài vạch, xe đỗ trái phép, quay đầu xe, đi xe đạp trên vỉa hè, v.v. (17 loại, xem bài báo gốc). Nhiệm vụ: chấm điểm bất thường cho từng khung hình.

## 2. Nhiệm vụ
Với mỗi khung hình của các video test, dự đoán **điểm bất thường** (số thực; càng cao càng bất thường). Một khung hình được tính là bất thường nếu có ít nhất một khung bao bất thường trong nhãn gốc.

## 3. Dữ liệu (gói `CCTV_data.zip`)
Các video được lấy mẫu **1 khung hình/giây**, thu nhỏ còn 320×180 (JPEG). Tên file là số thứ tự giây trong video, nên bạn có thể dùng khung hình lân cận (ngữ cảnh thời gian).

| Thư mục / file | Nội dung |
|---|---|
| `train/trNN/SSSSS.jpg` | 46 video **chỉ có cảnh bình thường** (không nhãn) |
| `validation/vaNN/SSSSS.jpg` + `validation_labels.csv` | 7 video có nhãn (`id,label`, label=1 là bất thường) để bạn tự đánh giá; id = `vaNN_SSSSS` |
| `test/teNN/SSSSS.jpg` | 28 video test, **không có nhãn**; id = `teNN_SSSSS` |
| `sample_submission.csv` | file nộp mẫu (điểm 0 cho mọi khung → AUROC 0.5) |
| `starter.py`, `scorer.py` | baseline nền cố định và hàm chấm (giống máy chấm) |

Tập test chia theo video: một nửa số video được chấm trong bảng public, nửa còn lại là private. Bạn không biết video nào thuộc phần nào.

**Giấy phép.** MERL Street Scene, © Mitsubishi Electric Research Laboratories, giấy phép **CC BY-SA 4.0**. Gói này là bản **thu nhỏ, lấy mẫu 1 khung hình/giây** (có chỉnh sửa) của dữ liệu gốc; vui lòng giữ nguyên giấy phép khi chia sẻ lại và trích dẫn: Ramachandra & Jones, *Street Scene: A new dataset and evaluation protocol for video anomaly detection*, WACV 2020. Nguồn: https://doi.org/10.5281/zenodo.10870472

## 4. File nộp
`submission.csv`, UTF-8, dòng đầu `id,score`, **đúng một dòng cho mỗi khung hình test** (xem `sample_submission.csv`). `score` là số hữu hạn bất kỳ.

## 5. Chấm điểm
- **Điểm chính: AUROC** của điểm bất thường so với nhãn trên các khung hình của tập chấm.
- Điểm phụ: AP (average precision).
- Máy chấm tính cả public và private; bảng xếp hạng hiện số public.

## 6. Luật
- Được dùng dữ liệu `train`, `validation` và mô hình/dữ liệu bên ngoài **có giấy phép cho phép**; ghi nguồn trong báo cáo.
- Không tìm cách lấy nhãn test từ dữ liệu gốc trên mạng (bộ Street Scene công khai), không gán nhãn thủ công cho test.
- Tối đa 20 lần nộp mỗi ngày.

## 7. Gợi ý
Mô hình tái dựng/dự đoán khung kế tiếp, khoảng cách đặc trưng (CLIP, DINOv2) tới tập cảnh bình thường, phát hiện chuyển động bất thường (hiệu khung, optical flow), làm mượt điểm theo thời gian, chuẩn hóa theo video.

## 8. Nguồn cảm hứng
Bài toán video anomaly detection giám sát (UCSD Ped, Avenue, ShanghaiTech, Street Scene); phong cách bài tập: IOAI 2026 (CC BY 4.0).
