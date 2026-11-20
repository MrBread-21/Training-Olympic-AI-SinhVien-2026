# Bài 26 (26/11) - Video: phân loại sự kiện + định vị đoạn khung

**Thời gian mục tiêu: ≤ 50 phút.**

## Đọc trước khi làm

Đọc trong khoảng 30-45 phút trước khi bấm giờ; ghi ra giấy 3-5 ý chính (công thức, tham số, bẫy). Lúc làm bài thì đóng hết.

- **OpenCV**: [cv::VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) (`read`, `grab`, `CAP_PROP_FRAME_COUNT`; số khung báo về có thể lệch, nên đọc tuần tự một lượt).
- **Bài báo**: TSN, Wang và cộng sự, *Temporal Segment Networks* [arXiv:1608.00859](https://arxiv.org/abs/1608.00859), mục 3.1-3.2 (lấy mẫu thưa N đoạn rải đều, gộp theo thời gian).
- **tIoU và chọn đoạn**: [README bài MSAD](https://github.com/MrBread-21/Training-Olympic-AI-SinhVien-2026/tree/MrBread/MSAD) trên GitHub (mục giải mã tối đa hoá tIoU kỳ vọng) và hàm `segment` trong `../../../challenges/MSAD/main.py`.
- **Trong máy**: `../../ON_TAP.md` phần 1.2 (phân loại video: cache khung, gộp thời gian, kênh chuyển động); `../../../challenges/MSAD/main.py` (đọc video 1 lượt, 48 vị trí, LR + TCN, XGB theo khung).

## Bài 1

- **Đầu vào**: `data/videos/<video_id>.mp4` (64x64, 40-64 khung, 12 fps, nền có vật thể trôi chậm); `data/train.csv` (`video_id,label,start_frame,end_frame`, 200 video); `data/video_info.csv` (số khung); `data/sample_submission.csv` (100 video test). 5 nhãn: `binh_thuong, nhay, quay, chay_ngang, nhap_nhay`.
- **Đầu ra**: `submission.csv` (`video_id,label,start_frame,end_frame`); khung đánh số từ 1, tính cả hai đầu; video `binh_thuong` để trống hai cột khung.
- **Metric** (giống MSAD): video bình thường được 1 nếu đoán bình thường; video có sự kiện được **tIoU** nếu đúng nhãn, sai nhãn được 0; trung bình theo lớp rồi trung bình đều. **Ngưỡng 0.50**.
- **Khung bắt buộc**:
  1. Đọc **N khung rải đều** (`np.linspace` theo số khung), đọc tuần tự 1 lượt.
  2. Đặc trưng từng khung bằng ResNet-18 đóng băng, **cache** `.npz`.
  3. Đặc trưng video = trung bình + độ lệch chuẩn theo thời gian + **trung bình |f(t) - f(t+k)|**, rồi logistic; báo OOF macro-F1.
  4. Định vị: logistic theo **từng khung** (nhãn khung = nằm trong đoạn sự kiện), làm mượt, chọn đoạn **tối đa hoá tIoU kỳ vọng**: max sum_trong(p) / (độ dài + sum_ngoài(p)), có tính số khung mỗi ô.
- Lời giải mẫu: 0.61 (OOF phân loại 0.68). Lớp `nhay` và `quay` hay nhầm nhau; xem ma trận nhầm lẫn rồi nghĩ đặc trưng gì tách được.

## Bài 2 - Viết lại nhanh bài 25 (≤ 20 phút)

Trong `d25_doi_khang`: `viet_lai.py` (PGD + Grad-CAM), chấm `python check_25.py viet_lai.py`.

## Cách chạy

```
python tao_du_lieu.py
python check_26.py bai_lam.py
```
