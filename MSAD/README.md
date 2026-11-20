# MSAD - Nhận diện và định vị sự kiện bất thường trong video giám sát

Bài luyện tập MSAD trên [hệ thống ôn luyện PHENIKAA AI Hackathon](https://aihackathon.phenikaa-uni.edu.vn/). Đề bài đầy đủ: [DE_BAI.md](DE_BAI.md).

## Bài toán

- Với mỗi video, dự đoán loại sự kiện (12 nhãn: `Normal`, `Assault`, `Explosion`, `Fighting`, `Fire`, `Object_falling`, `People_falling`, `Robbery`, `Shooting`, `Traffic_accident`, `Vandalism`, `Water_incident`) và đoạn khung `[start_frame, end_frame]` xảy ra sự kiện.
- 250 video có nhãn (train + validation), video test riêng. Phân bố lớp lệch (Normal nhiều nhất, Water_incident và Vandalism ít nhất). Video nhiều bối cảnh, độ phân giải và fps khác nhau.
- Điểm mỗi video: video Normal được 1 nếu đoán Normal; video bất thường được **tIoU khung hình** nếu đoán đúng nhãn, sai nhãn được 0. Điểm chính là trung bình theo từng lớp rồi trung bình đều 12 lớp.

Dữ liệu MSAD chỉ dùng cho nghiên cứu và luyện tập theo [thoả thuận dữ liệu](https://msad-dataset.github.io/agreement.html), nên repo không chứa video, khung hình hay nhãn.

**LB public tốt nhất: 0.4149**, đạt bằng bản không pretrained `main_scratch.py`.

Không xác định được lượt nộp nào đạt điểm này, nên repo không kèm file nộp.

## Hai lời giải

| File | Trọng số pretrained | OOF 5-fold | CV theo nhóm cảnh | Public |
|---|---|---|---|---|
| `main_scratch.py` | không (mọi mô hình train từ đầu) | 0.283 | 0.221 | **0.4149** (bản 2 mô hình bảng) |
| `main.py` (bản cuối) | backbone ImageNet của torchvision, **đóng băng** | **0.335** | **0.326** | 0.3745 (cấu hình trước bản cuối) |

"CV theo nhóm cảnh" gộp các video gần trùng nhau (cùng camera, cùng cảnh) vào cùng một fold, nên ít bị lạc quan hơn OOF thường.

### `main_scratch.py`: không dùng pretrained

1. **Đọc video một lần**, 48 cặp khung (t, t + 0.2 giây) lấy đều:
   - ảnh 128x72: 55 đặc trưng thủ công mỗi khung (optical flow Farneback: độ lớn, phân vị, hướng 8 ngăn, lưới 3x3, divergence; hiệu khung; histogram màu HSV, tỉ lệ pixel màu lửa và màu nước, độ sáng, cạnh; độ lệch so với nền tĩnh của video);
   - ảnh 112x112: RGB và (flow x, flow y, hiệu ảnh xám) cho CNN.
2. **Mô hình bảng**, 2 tập cột (A đầy đủ, B tập con): XGBoost + LogisticRegression + ExtraTrees trên thống kê theo thời gian để phân loại; XGBoost theo từng khung, có điều kiện lớp, để định vị.
3. **CNN two-stream** train từ đầu: ResNet + Squeeze-and-Excitation + Temporal Shift Module cho RGB, một luồng cho chuyển động, TCN theo thời gian; có đầu dự đoán từng khung và đầu dự đoán cả video; EMA trọng số.
4. **Ensemble**: p = 0.8 x (pA + pB)/2 + 0.2 x pCNN, nhân P(Normal) với 0.8; đoạn sự kiện là đoạn liên tục có tổng (p - 0.4) lớn nhất (thuật toán Kadane).

### `main.py`: backbone ImageNet đóng băng (bản cuối)

1. **Đọc video một lần** tại 48 vị trí: giữ nguyên 55 đặc trưng thủ công; khung t giữ tỉ lệ 16:9 (không ép vuông) và đi qua **ConvNeXt-tiny ở 640x360** và **ResNet-50 ở 448x252**, lấy GAP của bản đồ đặc trưng cuối. CPU đọc video song song, GPU chạy backbone.
2. **Phân loại** = 0.5 x LogisticRegression (C = 0.0005, cân bằng lớp) trên mean/max/std theo thời gian + 0.5 x **đầu TCN** (dilation 1/2/4, attention pooling và max pooling; 5 seed) trên chuỗi đặc trưng backbone, rồi nhân P(Normal) với 0.8. Đầu TCN học cùng lúc nhãn video và nhãn từng khung.
3. **Định vị** = XGBoost theo khung có điều kiện lớp trên đặc trưng thủ công (2 tập cột), làm mượt, rồi chọn đoạn **tối đa hoá tIoU kỳ vọng**: với xác suất khung p, chọn đoạn S tối đa sum(p trong S) / (|S| + sum(p ngoài S)), thay cho Kadane có ngưỡng.

Chạy khoảng 16 phút trên GTX 1650, gồm cả đọc video.

## Diễn biến của `main.py` (OOF / CV theo nhóm cảnh)

| Thay đổi | OOF | CV nhóm | Public |
|---|---|---|---|
| Khung 224x224, định vị Kadane | 0.311 | 0.301 | 0.300 |
| Giải mã tối đa hoá tIoU kỳ vọng | 0.313 | 0.308 | - |
| Khung 448x252 giữ tỉ lệ, có TTA lật | 0.330 | 0.322 | - |
| Khung 448x252, không lật | 0.334 | 0.314 | 0.3745 |
| **ConvNeXt 640x360 + ResNet-50 448x252** (bản cuối) | **0.335** | **0.326** | - |
| Cả hai backbone ở 640 | 0.326 | 0.319 | - |
| Trung bình 2 mô hình ở 448 và 640 (trích đặc trưng khoảng 28 phút, quá giới hạn 20 phút) | 0.341 | 0.325 | - |

## Đã thử và không giữ

Đều không vượt mức nhiễu trên CV theo nhóm cảnh:

- **Đầu phân loại**: Transformer 2/4/6 lớp (0.254 khi chỉ dùng đầu, so với TCN 0.288); TCN 6 lớp; TCN + Transformer; 160 epoch; mixup (alpha 0.2/0.4/1.0); ghép thêm đặc trưng thủ công vào TCN. Dò trọng số ensemble kiểu Caruana chỉ giữ LR + TCN.
- **Đặc trưng**: spatial max pooling, EfficientNet-B0, embedding chuyển động, LR theo từng khung, đặc trưng "mới lạ" cho định vị.
- **Truy hồi**: GeM, VLAD (NetVLAD không giám sát), kNN với query expansion; đối sánh patch cho định vị chỉ +0.003.
- **Fine-tune một phần backbone** (stage cuối của ConvNeXt kiểu TSN, 112/224px): 0.22-0.23, kém hẳn bản đóng băng.
- **Hiệu chỉnh**: hệ số nhân theo từng lớp bị quá khớp (-0.027 khi chấm trên nửa giữ lại); ngưỡng theo lớp; kNN ghi đè nhãn; pseudo-label val và test chỉ +0.003.
- **Các thử nghiệm đầu**: GBM trên đặc trưng chuyển động và màu đạt public 0.363, trong khi ViT/DINOv2 train từ đầu chỉ 0.114. Với khoảng 250 video, đặc trưng thủ công tốt hơn hẳn mô hình lớn train từ đầu.

## Bài học

- **Video ít thì đặc trưng tốt hơn mô hình end-to-end**: backbone đóng băng cộng bộ phân loại nhỏ ổn định hơn fine-tune; không có pretrained thì đặc trưng chuyển động và màu thủ công vẫn rất mạnh.
- **CV phải chia theo nhóm cảnh**: nhiều video quay cùng một camera, nên CV ngẫu nhiên đánh giá quá cao.
- **Giải mã theo đúng metric**: chọn đoạn tối đa hoá tIoU kỳ vọng thay cho ngưỡng cố định giúp tăng điểm mà không phải chỉnh tham số.
- **Public chỉ khoảng 40% video test** nên rất nhiễu: bản không pretrained có OOF thấp hơn nhiều nhưng public lại cao hơn (0.4149 so với 0.3745).

## Chạy

Đặt `videos/`, `train_labels.csv`, `validation_labels.csv`, `test_videos.csv`, `video_info.csv` và `scorer.py` của bài cùng thư mục với code:

```
python main.py                 # bản cuối, ghi submission.csv
python main.py --cv --reps 3   # đánh giá OOF
python main_scratch.py         # bản không dùng pretrained
```

Đặc trưng được cache vào `outputs/` sau lần chạy đầu. Thư viện: `torch`, `torchvision`, `opencv-python`, `numpy`, `scikit-learn`, `xgboost`.

## Tham khảo

- Zhu, L., Wang, L., Raj, A., Gedeon, T., Chen, C. (2024). *Advancing Video Anomaly Detection: A Concise Review and a New Dataset*. NeurIPS Datasets and Benchmarks. [github.com/Tom-roujiang/MSAD](https://github.com/Tom-roujiang/MSAD).
- Liu, Z. và cộng sự (2022). *A ConvNet for the 2020s* (ConvNeXt). [arXiv:2201.03545](https://arxiv.org/abs/2201.03545).
- He, K. và cộng sự (2016). *Deep Residual Learning for Image Recognition*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
- Bai, S., Kolter, J. Z., Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* (TCN). [arXiv:1803.01271](https://arxiv.org/abs/1803.01271).
- Lin, J., Gan, C., Han, S. (2019). *TSM: Temporal Shift Module for Efficient Video Understanding*. [arXiv:1811.08383](https://arxiv.org/abs/1811.08383).
- Hu, J., Shen, L., Sun, G. (2018). *Squeeze-and-Excitation Networks*. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507).
- Chen, T., Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754).
- Caruana, R. và cộng sự (2004). *Ensemble Selection from Libraries of Models*. ICML.
- Farnebäck, G. (2003). *Two-Frame Motion Estimation Based on Polynomial Expansion*. SCIA.
