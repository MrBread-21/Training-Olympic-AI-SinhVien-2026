# Bài luyện tập MSAD · Nhận diện và định vị sự kiện bất thường trong video giám sát

*Bài ôn luyện nội bộ cho đội tuyển (nghiên cứu và luyện tập OLP). Dữ liệu: MSAD (Multi-Scenario Anomaly Detection). **Không chia sẻ dữ liệu ra ngoài đội.***

## 1. Bối cảnh
Camera giám sát ghi hàng giờ video mỗi ngày. Một hệ thống hữu ích phải trả lời được hai câu hỏi: **chuyện gì đã xảy ra** (đánh nhau, cháy, tai nạn giao thông, người ngã, …) và **xảy ra từ khung hình nào đến khung hình nào**. Bài toán kết hợp nhận dạng hành động/sự kiện trong video với định vị theo thời gian.

## 2. Bài toán
Với mỗi video test, dự đoán:
1. **`label`**: loại sự kiện của video, một trong 12 nhãn:
   `Normal, Assault, Explosion, Fighting, Fire, Object_falling, People_falling, Robbery, Shooting, Traffic_accident, Vandalism, Water_incident`;
2. **`start_frame`, `end_frame`**: khoảng khung hình xảy ra sự kiện bất thường, **đánh số từ 1, tính cả hai đầu** (khung hình đầu tiên của video là 1, khung cuối là `frames` trong `video_info.csv`).

Mỗi video bất thường có đúng một khoảng bất thường. Video `Normal` không có sự kiện: để trống hai cột khung hình.

## 3. Dữ liệu
Trên trang bài có các nút tải (cần đăng nhập): `MSAD_data.zip` (nhãn, đề, `starter.py`, `scorer.py`), `videos_train.zip`, `videos_validation.zip`, `videos_test.zip`. Giải nén video vào cùng thư mục `videos/` cạnh các tệp `.csv`.

| tệp | nội dung |
|---|---|
| `train_labels.csv` | `video_id,label,start_frame,end_frame` của các video train |
| `validation_labels.csv` | cùng định dạng, để tự đánh giá |
| `test_videos.csv` | danh sách `video_id` cần dự đoán (không có nhãn) |
| `video_info.csv` | `video_id, split, frames, fps, width, height` |
| `classes.csv` | 12 nhãn hợp lệ |
| `sample_submission.csv` | bài nộp hợp lệ mẫu (mọi video là `Normal`) |

- Video có nhiều bối cảnh (cửa hàng, đường phố, nhà xưởng, bờ nước, …), độ phân giải và tốc độ khung hình khác nhau; số khung hình và fps lấy từ `video_info.csv` (đếm bằng OpenCV `CAP_PROP_FRAME_COUNT`).
- Mã video là ngẫu nhiên; video đã bỏ âm thanh và metadata. Ba tập chia theo video, mỗi loại sự kiện đều có mặt ở cả ba tập.
- Phân bố lớp **không đều** (Normal nhiều nhất; Water_incident, Vandalism ít nhất).

## 4. Định dạng bài nộp
`submission.csv`, UTF-8, tiêu đề đúng `video_id,label,start_frame,end_frame`, **đúng một dòng cho mỗi video** trong `test_videos.csv`:

```
video_id,label,start_frame,end_frame
V0007,Fire,35,410
V0012,Normal,,
V0019,People_falling,120,188
```

Bài bị từ chối (không tính lượt) nếu sai tiêu đề/số cột, thiếu/trùng/thừa video, nhãn không thuộc 12 nhãn, hoặc với nhãn khác `Normal` mà khung hình không phải số nguyên thỏa `1 ≤ start_frame ≤ end_frame ≤ frames`.

## 5. Chấm điểm
Điểm của từng video:
- video `Normal`: **1** nếu dự đoán `Normal`, ngược lại **0**;
- video thuộc loại bất thường $c$: **tIoU khung hình** nếu dự đoán đúng nhãn $c$, ngược lại **0**, với

$$\mathrm{tIoU} = \frac{|[\hat{s}, \hat{e}] \cap [s, e]|}{|[\hat{s}, \hat{e}] \cup [s, e]|} \quad (\text{đếm số khung hình, tính cả hai đầu}).$$

**Điểm chính** = trung bình điểm video **trong từng lớp**, rồi trung bình đều trên 12 lớp (macro), thang 0–1. Như vậy đoán sai loại sự kiện thì không được điểm khung hình, và lớp ít video cũng quan trọng như lớp nhiều video.

Bảng xếp hạng còn hiện: **macro-F1** của phân loại và **mIoU khung hình** trên các video bất thường (không xét nhãn), để bạn biết mình yếu ở phần nào. Bảng hiện phần **public** (khoảng 40% video test, chia theo lớp); phần còn lại là private. `scorer.py` là đúng hàm chấm của máy chủ.

## 6. Hướng giải gợi ý
- Trích đặc trưng khung hình bằng mô hình ảnh (CLIP/SigLIP, DINOv2) hoặc video (VideoMAE, X-CLIP), gộp theo thời gian để phân loại; có thể dùng zero-shot với tên lớp làm câu mô tả.
- Định vị: chấm điểm "bất thường" theo từng đoạn (độ lệch so với đầu video, độ tương đồng với mô tả lớp, năng lượng chuyển động), làm mượt rồi chọn đoạn liên tục có điểm cao; học ngưỡng/độ dài theo từng lớp trên train.
- Mô hình mã nguồn mở có giấy phép cho phép; ghi nguồn mô hình trong báo cáo.

## 7. Luật luyện tập
1. Không gán nhãn tay, không xem video test để viết luật riêng cho từng video.
2. **Không tra cứu đáp án**: không dùng bản MSAD gốc, chú thích MSAD/VALU công khai hay so khớp video test với bộ dữ liệu gốc. MSAD là bộ công khai nên việc tra cứu là có thể về kỹ thuật, nhưng làm vậy thì điểm không còn ý nghĩa luyện tập.
3. Tối đa 20 lần nộp mỗi ngày.

## 8. Quyền riêng tư và giấy phép
- Video là cảnh quay giám sát có thể chứa người thật. Chỉ dùng cho nghiên cứu và luyện tập của đội tuyển; **không phát tán video, không đăng khung hình lên mạng, không cố nhận dạng người trong video**. Xóa dữ liệu khi không còn dùng.
- Dữ liệu dùng theo thỏa thuận dữ liệu MSAD (https://msad-dataset.github.io/agreement.html), BTC đã được cấp quyền cho nghiên cứu và luyện tập.
- Trích dẫn: Zhu, Wang, Raj, Gedeon, Chen. *Advancing Video Anomaly Detection: A Concise Review and a New Dataset.* NeurIPS 2024. https://github.com/Tom-roujiang/MSAD
