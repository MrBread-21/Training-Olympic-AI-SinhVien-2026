# CCTV Street Scene - Phát hiện bất thường trên camera giám sát

Bài luyện tập CCTV trên [hệ thống ôn luyện PHENIKAA AI Hackathon](https://aihackathon.phenikaa-uni.edu.vn/). Đề bài đầy đủ: [DE_BAI.md](DE_BAI.md).

## Bài toán

- Một camera cố định nhìn xuống đoạn phố hai làn xe. Dữ liệu lấy mẫu 1 khung/giây, ảnh 320x180.
- Train: 46 video **chỉ có cảnh bình thường**, không nhãn. Val: 7 video có nhãn từng khung. Test: 28 video.
- Chấm điểm bất thường cho từng khung. Điểm: **AUROC gộp mọi khung test** (phụ: AP). Bất thường gồm đi sai làn, băng qua đường ngoài vạch, đỗ xe trái phép, quay đầu, đi xe đạp trên vỉa hè...

**LB public tốt nhất: 0.6761.** `solve.py` với cấu hình mặc định tái tạo đúng lượt nộp LB 0.6727.

Không xác định được lượt nộp đạt 0.6761. Thư mục `submissions/` có 2 lượt của `solve.py` đã biết điểm: [a = 2, LB 0.6727](submissions/submission_solve_a2_lb0.6727.csv) và [a = 1.5, LB 0.6729](submissions/submission_solve_a1.5_lb0.6729.csv).

## Phát hiện quan trọng nhất (EDA)

1. **Tỉ lệ khung bất thường chênh nhau rất nhiều giữa các video** (va03 là 5%, va07 là 83%). AUROC tính gộp mọi khung của mọi video, nên chênh lệch *giữa các video* chiếm phần lớn tín hiệu. Chuẩn hoá điểm theo từng video sẽ kéo AUROC về khoảng 0.5.
2. **Mọi detector kiểu "khác cảnh bình thường" đều chạy ngược chiều khi gộp** (val 0.25-0.30): kNN đặc trưng CNN, mô hình đoán ô bị che, độ khác nền. Chúng thực chất đo mật độ giao thông và ánh sáng, trong khi các cảnh bất thường được dàn dựng lúc đường vắng. Chỉ riêng việc đổi dấu điểm "khác nền trung vị" của baseline đã cho val 0.743.
3. Trần lý thuyết trên val: kể cả biết đúng tỉ lệ bất thường của từng video thì AUROC gộp cũng chỉ đạt 0.85. Phần còn lại phải đến từ việc xếp hạng đúng *bên trong* mỗi video.

## Lời giải cuối: `solve.py`

Chạy trên CPU khoảng 1.5 phút, không dùng trọng số pretrained. Điểm mỗi khung là tổng của hai phần:

**điểm = z(trong video) + a x z(mức video)**, mặc định `a = 2`.

- **Mức video**: `-log(chuyển động trung bình của cả video)`, chuyển động là trung bình |khung t - khung t-1| trên ảnh xám 80x45. Đường càng vắng càng dễ có cảnh bất thường. Chuẩn hoá z qua các video val và test, không dùng nhãn. Riêng phần này: val gộp 0.834, LB 0.627.
- **Trong video** (không dùng nhãn):
  1. Tách blob bằng trừ nền, hai cách: nền là trung vị của chính video đó (ảnh 320x180), hoặc nền chung của train sau khi chuẩn hoá tương phản cục bộ (LCN, giảm ảnh hưởng của ánh sáng) với ngưỡng riêng cho từng pixel (ảnh 160x90).
  2. Đặc trưng mỗi blob: tâm (x, y), log diện tích, tỉ lệ cao/rộng, chuyển động trong khung bao, khoảng cách tới blob gần nhất ở khung trước, hướng dịch chuyển (dx, dy), tỉ lệ pixel đứng yên.
  3. Điểm blob là khoảng cách trung bình tới 5 blob gần nhất trong 46 video train (đặc trưng chuẩn hoá theo train, **vị trí nhân trọng số 3-5** vì làn đường và vỉa hè là manh mối chính). Điểm khung là max (hoặc tổng) điểm blob.
  4. Trung bình 7 biến thể (khác nhau ở cách tách nền, trọng số vị trí, cách gộp, tập đặc trưng), chuẩn hoá z theo video, làm mượt Gaussian sigma = 3.

| Phần | val |
|---|---|
| Trong video, AUROC trung bình 7 video | 0.663 |
| Chỉ mức video, AUROC gộp | 0.834 |
| Tổng, AUROC gộp | 0.813 (AP 0.596) |

`solve.py` chạy lại cho ra file giống hệt file đã nộp (LB 0.6727).

## Diễn biến điểm LB public

| Cách làm | LB public |
|---|---|
| Baseline: khác nền trung vị, đổi dấu | 0.6400 |
| Hướng vật thể (`solution.ipynb`, bản 1): ghép baseline + 2 lần nhánh vị trí H4 | 0.6639 |
| Hướng vật thể bản 2 (che nền trong ống vật thể) | 0.6018 |
| kNN patch ResNet-18 ImageNet (kiểu PatchCore) | 0.53-0.55 |
| Transformer đoán ô bị che trên đặc trưng ResNet-50 ImageNet | 0.52 |
| Chỉ phần mức video | 0.6271 |
| Mức video + LR có giám sát học từ 7 video val (ResNet-50 layer2+3) | 0.6475 |
| Như trên, ensemble 3 bộ đặc trưng (ResNet-50 layer3 GAP và max, ConvNeXt-tiny) | 0.6453 |
| Mức video + blob kNN 1 biến thể, a = 3 | 0.6671 |
| `solve.py`: blob kNN 7 biến thể, a = 3 | 0.6671 |
| **`solve.py`, a = 2 (mặc định)** | **0.6727** |
| `solve.py`, a = 1.5 | 0.6729 |
| **Lượt nộp tốt nhất** (cấu hình không được ghi lại trong log) | **0.6761** |

Lúc đó đội dẫn đầu bài có LB 0.6978.

## Hướng thử nghiệm: `solution.ipynb` (vật thể, tự giám sát)

Ý tưởng: bất thường nằm ở từng vật thể, nên học "vật thể bình thường trông thế nào, ở đâu, di chuyển ra sao" chỉ từ video train. Không dùng trọng số pretrained, train khoảng 20 phút trên GTX 1650.

1. **Tách vật thể** bằng OpenCV: khác nền trung vị, connected components, optical flow Farneback. Cắt "ống vật thể" 5 khung (t-2 đến t+2) kích thước 32x32 kèm flow.
2. **Mạng đa nhiệm** (encoder ResNet nhỏ, bộ nhớ kiểu MemAE), 5 nhánh: H1 tái tạo khung t qua bộ nhớ, H2 đoán ống xuôi hay đảo thời gian, H3 đoán khung liên tiếp hay nhảy cóc, H4 đoán ô lưới 6x10 nơi vật thể đứng, H5 tái tạo optical flow.
3. Chuẩn hoá lỗi từng nhánh theo vật thể của 2 video train để riêng; điểm khung là max theo vật thể, làm mượt theo thời gian; ghép với baseline bằng rank-average.

Notebook hiện là bản 2 (che nền trong ống vật thể để mạng "bớt học thuộc nền"). Thay đổi này làm LB **giảm** từ 0.6639 xuống 0.6018: camera cố định và test cùng khung cảnh, nên nền chính là manh mối vị trí đúng. H2 và H3 gần như không học được ở 1 khung/giây vì vật thể dịch quá xa giữa hai khung.

## Bài học

1. **Xem thống kê nhãn theo nhóm (video) trước khi làm mô hình**, và kiểm tra **chiều của điểm** (có cần đổi dấu không).
2. **Val ít nhóm thì không đáng tin để chọn cấu hình**: chỉ 7 video, nhiều lần val tăng nhưng LB giảm. LR có giám sát học trên val cải thiện LOVO từ 0.66 lên 0.787 nhưng LB không đổi, vì nó khớp theo sự kiện riêng của 7 video val. Trọng số `a` của phần mức video cũng vậy: val muốn `a` lớn (a = 5 cho val 0.845) vì val đánh giá phần này quá cao (val 0.834, LB 0.627), còn LB thì muốn `a` nhỏ.
3. **Tín hiệu không nhãn chuyển sang test tốt hơn tín hiệu có giám sát học từ ít nhóm**: blob kNN không dùng nhãn (0.6671) hơn LR có giám sát (0.645).
4. **Tăng điểm bằng tín hiệu mới khác bản chất**, không bằng trộn lại tín hiệu cũ. Ghép thêm tín hiệu yếu luôn kéo điểm xuống.
5. Đừng sửa thứ đang chạy tốt trên LB chỉ dựa vào lập luận "về lý thuyết" (bài học che nền).

## Chạy

Đặt `train/`, `validation/`, `test/`, `validation_labels.csv`, `sample_submission.csv` cùng thư mục:

```
python solve.py                    # lời giải cuối, ghi submission.csv
python solve.py --a 1.5            # trọng số phần mức video khác
jupyter nbconvert --to notebook --execute solution.ipynb    # hướng vật thể (cần GPU)
```

Thư viện: `numpy`, `pandas`, `opencv-python`, `scipy`, `scikit-learn`; notebook cần thêm `torch`, `matplotlib`.

## Tham khảo

- Ramachandra, B., Jones, M. J. (2020). *Street Scene: A new dataset and evaluation protocol for video anomaly detection*. WACV. Dữ liệu: [doi:10.5281/zenodo.10870472](https://doi.org/10.5281/zenodo.10870472) (CC BY-SA 4.0).
- Georgescu, M.-I. và cộng sự (2021). *Anomaly Detection in Video via Self-Supervised and Multi-Task Learning*. CVPR. [arXiv:2011.07491](https://arxiv.org/abs/2011.07491).
- Gong, D. và cộng sự (2019). *Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection* (MemAE). ICCV. [arXiv:1904.02639](https://arxiv.org/abs/1904.02639).
- Roth, K. và cộng sự (2022). *Towards Total Recall in Industrial Anomaly Detection* (PatchCore). CVPR. [arXiv:2106.08265](https://arxiv.org/abs/2106.08265).
- Farnebäck, G. (2003). *Two-Frame Motion Estimation Based on Polynomial Expansion*. SCIA.
