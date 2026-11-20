# ImgCap - Mô tả ảnh (image captioning)

Bài luyện tập CAPTION trên [hệ thống ôn luyện PHENIKAA AI Hackathon](https://aihackathon.phenikaa-uni.edu.vn/). Đề bài đầy đủ: [DE_BAI.md](DE_BAI.md).

## Bài toán

- Sinh một câu mô tả tiếng Anh cho mỗi ảnh.
- Train: 3500 ảnh COCO, mỗi ảnh 5 câu mô tả. Val: 300 ảnh COCO. Test: 800 ảnh **nocaps** (ảnh Open Images, nhiều vật thể và bối cảnh không có trong COCO).
- Điểm: CIDEr-D (phụ: BLEU-4, ROUGE-L).

**LB public tốt nhất: 1.1927**, đạt bằng bản BLIP ở mục "Phiên bản trước". Bản không pretrained trong repo không có ghi chép LB riêng.

File nộp đạt điểm này: [submissions/submission_blip_lb1.1927.csv](submissions/submission_blip_lb1.1927.csv).

Lời giải trong `solution.ipynb` **không dùng trọng số pretrained** (chỉ dùng kiến trúc, trọng số tự train), theo luật của kỳ thi OlpAI mà đội đang luyện.

## Ý tưởng

Với 3500 ảnh, một mô hình sinh câu CNN + LSTM/Transformer train từ đầu sẽ rất yếu (câu sai ngữ pháp, lặp từ). Vì vậy lời giải dùng **truy hồi và đồng thuận** (nearest-neighbor captioning): mượn nguyên câu của các ảnh train giống nhất, rồi chọn câu "trung tâm" nhất.

1. **Học đặc trưng ảnh bằng nhãn tự tạo**: lấy 400 "từ khái niệm" hay gặp trong caption train (bỏ stopword, từ xuất hiện ở ít nhất 5 ảnh), mỗi ảnh có nhãn multi-label là các từ có trong 5 câu của nó (trung bình 11.3 từ/ảnh). Train **ResNet-18 khởi tạo ngẫu nhiên** dự đoán các từ này bằng BCE: ảnh 128px, RandomResizedCrop, lật ngang, ColorJitter, AdamW lr 2e-3, OneCycle, số epoch tự tính theo ngân sách 10 phút (13 epoch trên GTX 1650).
2. **Tìm láng giềng**: đặc trưng là lớp trước `fc` (trung bình ảnh gốc và ảnh lật, chuẩn hoá L2), lấy `K` ảnh train có cosine cao nhất.
3. **Chọn câu đồng thuận (MBR)**: gom 5K câu của các láng giềng, chọn câu có tổng độ tương đồng CIDEr-D lớn nhất với các câu còn lại, mỗi câu đóng góp theo độ giống ảnh của láng giềng chứa nó. IDF tính trên caption train. `K` chọn trên val.

## Kết quả (val CIDEr-D, 300 ảnh COCO)

| Cách làm | val CIDEr-D |
|---|---|
| Baseline `starter.py` (truy hồi theo điểm ảnh 16x16) | 0.0686 |
| Láng giềng gần nhất, lấy câu đầu tiên | 0.1126 |
| MBR trên 5 láng giềng | 0.1666 |
| MBR trên 10 láng giềng | 0.1987 |
| **MBR trên 20 láng giềng** | **0.2725** |

Cả notebook chạy khoảng 19 phút trên GTX 1650. Lần chạy này không có ghi chép điểm LB.

## Phiên bản trước (dùng pretrained, code không còn giữ)

Trước khi chuyển sang luật không dùng pretrained, đội dùng `Salesforce/blip-image-captioning-large` (đã tinh chỉnh trên COCO):

| Cách làm | val CIDEr-D | LB |
|---|---|---|
| BLIP-large, prompt "a picture of", beam 3, `max_new_tokens=20`, `min_length=5`, `repetition_penalty=1.3`, hậu xử lý bỏ "arafed", "there is", từ lặp liền nhau | 1.503 | **1.1927** |
| Như trên, chọn câu bằng MBR trong các câu beam | 1.508 | - |

- Beam 5 kém hơn beam 3 (1.477 so với 1.503), vì CIDEr phạt câu có độ dài lệch so với câu tham chiếu.
- Val (COCO) đánh giá cao hơn test (nocaps) khoảng 20%.

So với BLIP, lời giải không pretrained thấp hơn rất nhiều. Đây là giới hạn của việc chỉ có 3500 ảnh để học biểu diễn ảnh.

## Chạy

Đặt dữ liệu (`train/`, `val/`, `test/`, `train_captions.json`, `val_captions.json`) cùng thư mục với notebook:

```
jupyter nbconvert --to notebook --execute solution.ipynb
```

Thư viện: `torch`, `torchvision`, `numpy`, `pandas`, `Pillow`, `matplotlib`, `tqdm`. Hàm chấm CIDEr-D được viết lại trong notebook, giống `scorer.py` của bài.

## Tham khảo

- Vedantam, R., Zitnick, C. L., Parikh, D. (2015). *CIDEr: Consensus-based Image Description Evaluation*. [arXiv:1411.5726](https://arxiv.org/abs/1411.5726).
- Devlin, J. và cộng sự (2015). *Exploring Nearest Neighbor Approaches for Image Captioning*. [arXiv:1505.04467](https://arxiv.org/abs/1505.04467).
- Kumar, S., Byrne, W. (2004). *Minimum Bayes-Risk Decoding for Statistical Machine Translation*. HLT-NAACL.
- Chen, X. và cộng sự (2015). *Microsoft COCO Captions: Data Collection and Evaluation Server*. [arXiv:1504.00325](https://arxiv.org/abs/1504.00325).
- Agrawal, H. và cộng sự (2019). *nocaps: novel object captioning at scale*. ICCV. [arXiv:1812.08658](https://arxiv.org/abs/1812.08658).
- Li, J. và cộng sự (2022). *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation*. [arXiv:2201.12086](https://arxiv.org/abs/2201.12086).
- He, K. và cộng sự (2016). *Deep Residual Learning for Image Recognition*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
