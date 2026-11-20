# Training Olympic AI Sinh viên 2026

Repo luyện tập của đội cho kỳ thi Olympic Trí tuệ nhân tạo sinh viên Việt Nam 2026 (OlpAI'26). Mỗi thành viên làm bài trên **nhánh riêng**; nhánh `main` chỉ giữ bản mô tả này.

## Nội dung luyện tập

- Lời giải các bài trên [hệ thống ôn luyện PHENIKAA AI Hackathon](https://aihackathon.phenikaa-uni.edu.vn/): phân loại ảnh, mô tả ảnh, hỏi đáp tiếng Việt, phát hiện bất thường trong video giám sát.
- Bài tập NLP tổng hợp từ các cuộc thi Kaggle, DrivenData và bộ dữ liệu nghiên cứu.
- Luyện theo luật của kỳ thi: phần lớn lời giải **không dùng trọng số pretrained** (chỉ dùng kiến trúc, tự train) hoặc chỉ dùng backbone ImageNet được cấp, chạy offline trên GPU nhỏ trong khoảng 20 phút.

## Các nhánh

| Nhánh | Nội dung |
|---|---|
| [`MrBread`](../../tree/MrBread) | Lời giải 5 bài ôn luyện PHENIKAA AI Hackathon, mỗi bài có đề, code, README mô tả cách giải, số liệu và bài học |
| [`feature/hue`](../../tree/feature/hue) | Bộ 16 bài luyện NLP (8 nhóm dạng bài, mỗi nhóm 2 bài) |

### Nhánh `MrBread`

| Bài | Nhiệm vụ | Hướng giải trong repo (không pretrained) | LB public tốt nhất |
|---|---|---|---|
| EuroSAT | Phân loại ảnh vệ tinh 10 lớp khi ít nhãn, ảnh test bị làm hỏng | Augmentation mô phỏng biến đổi của test, chọn model trên val bị làm hỏng, ResNet-18 train từ đầu + pseudo-label, TTA 8 phép xoay/lật | 0.8074 (bản fine-tune ResNet-50 pretrained) |
| ImgCap | Mô tả ảnh (train COCO, test nocaps) | ResNet-18 train từ đầu dự đoán từ khái niệm, truy hồi ảnh gần nhất, chọn câu đồng thuận (MBR theo CIDEr) | 1.1927 (bản BLIP pretrained) |
| VietnameseExtract | Hỏi đáp trích xuất tiếng Việt | Sinh span ứng viên, xếp hạng bằng XGBRanker | 0.6517 (bản XLM-R pretrained) |
| CCTV Street Scene | Điểm bất thường từng khung, camera cố định | Điểm mức video (đường vắng) + kNN của blob vật thể, chạy CPU | 0.6761 |
| MSAD | Phân loại và định vị sự kiện bất thường trong video | Đặc trưng chuyển động/màu thủ công + mô hình bảng và CNN train từ đầu; thêm bản backbone ImageNet đóng băng + LR + TCN | 0.4149 (bản không pretrained) |

Với EuroSAT, ImgCap và VietnameseExtract, điểm LB tốt nhất đến từ các bản dùng mô hình pretrained viết trước khi chuyển sang luật không pretrained; code các bản đó không còn giữ.
Chi tiết từng bài (dữ liệu cần chuẩn bị, cách chạy, phân tích lỗi, nguồn tham khảo) nằm trong README của từng thư mục trên nhánh tương ứng.

## Quy ước

- Mỗi thành viên làm trên nhánh của mình, không đẩy thẳng lên `main`.
- Không đưa dữ liệu của ban tổ chức, trọng số mô hình và file cache lên repo; README mỗi bài ghi rõ cần đặt dữ liệu ở đâu.
- Mỗi bài có README nêu: bài toán, cách giải, số liệu (val và LB), những gì đã thử, bài học.
