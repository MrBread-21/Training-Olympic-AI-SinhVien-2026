# Lời giải các bài ôn luyện PHENIKAA AI Hackathon (nhánh MrBread)

Lời giải của MrBread cho các bài trên [hệ thống ôn luyện PHENIKAA AI Hackathon](https://aihackathon.phenikaa-uni.edu.vn/), trong quá trình chuẩn bị cho kỳ thi Olympic AI sinh viên 2026.

Phần lớn lời giải được viết theo luật của kỳ thi: **không dùng trọng số pretrained** (chỉ dùng kiến trúc, trọng số tự train trên dữ liệu được cấp), hoặc chỉ dùng backbone ImageNet trong danh sách được cấp và để đóng băng. Hầu hết chạy được trên GPU 4 GB (GTX 1650, fp32) trong khoảng 20 phút.

| Bài | Nhiệm vụ | Hướng giải chính | Kết quả |
|---|---|---|---|
| [EuroSAT](EuroSAT/) | Phân loại ảnh vệ tinh 10 lớp khi ít nhãn, ảnh test bị làm hỏng | Augmentation mô phỏng biến đổi của test, chọn model trên val bị làm hỏng, ResNet-18 train từ đầu, pseudo-label, TTA 8 phép xoay/lật | val hard macro-F1 0.665 (không pretrained) |
| [ImgCap](ImgCap/) | Mô tả ảnh (COCO, test nocaps) | ResNet-18 train từ đầu dự đoán 400 từ khái niệm, truy hồi 20 ảnh gần nhất, chọn câu đồng thuận (MBR theo CIDEr) | val CIDEr-D 0.2725 (starter 0.069) |
| [VietnameseExtract](VietnameseExtract/) | Hỏi đáp trích xuất tiếng Việt | Chọn câu theo IDF, sinh span ứng viên, xếp hạng bằng XGBRanker, thêm SQuAD tiếng Anh qua đặc trưng không phụ thuộc ngôn ngữ | val Jaccard 0.227 (starter 0.095) |
| [CCTV_Street_Scence](CCTV_Street_Scence/) | Điểm bất thường từng khung, camera giám sát cố định | Điểm mức video (đường vắng) + kNN của blob vật thể so với video bình thường, chạy CPU | **LB 0.6727** |
| [MSAD](MSAD/) | Phân loại và định vị sự kiện bất thường trong video | Đặc trưng chuyển động/màu thủ công + backbone ImageNet đóng băng, LR + TCN, XGBoost định vị theo tIoU kỳ vọng | OOF 0.335, public 0.415 (bản không pretrained) |

Mỗi thư mục có README riêng mô tả bài toán, cách giải chi tiết, số liệu, các hướng đã thử và bài học.

## Lưu ý

- Repo **không chứa dữ liệu**, trọng số và file nộp. Đặt dữ liệu của từng bài vào đúng thư mục bài trước khi chạy (xem mục "Chạy" trong README của bài).
- Một số bài cần `scorer.py` đi kèm gói dữ liệu của bài.
- Điểm "val" là điểm tự đánh giá trên tập validation; điểm "LB"/"public" là điểm trên bảng xếp hạng public của hệ thống ôn luyện.
