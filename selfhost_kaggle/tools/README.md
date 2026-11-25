# Dựng lại bộ đề selfhost_kaggle

Script tải dữ liệu gốc, dựng gói đề (dữ liệu cho thí sinh và đáp án), kiểm tra metric, chạy baseline và nén gói để tải lên Kaggle. Mọi thứ sinh ra (`raw/`, `<Bài>/upload/`, `<Bài>/host/`, `*_data.zip`) đều bị `.gitignore` chặn.

## Seed

Seed quyết định id ngẫu nhiên, cách chia Public/Private, nhãn nhiễu, ảnh bị làm hỏng, nên quyết định toàn bộ đáp án. Seed đặt qua biến môi trường `SELFHOST_SEED`, không ghi vào code. Buổi thi 26/11/2026 dùng `SELFHOST_SEED=2026`; buổi mới nên đổi seed để đề không trùng.

## Các bước

```
cd selfhost_kaggle
python tools/tai_du_lieu.py                       # khoảng 24 GB vào raw/, Nordland chiếm phần lớn (dimabsa: 3 MB)
python tools/giai_nen_nordland.py winter          # jpg 640x360, mỗi mùa khoảng 15 phút
python tools/giai_nen_nordland.py summer
SELFHOST_SEED=2026 python tools/xay_de.py         # dựng cả 13 bài (khoảng 1 giờ, VOC và Nordland lâu nhất)
python tools/kiem_tra_metric.py Nordland_PlaceRecognition VSFC_Sentiment ...
bash tools/chay_baseline.sh Nordland_PlaceRecognition VSFC_Sentiment ...
python tools/dong_goi.py                          # <Bài>/<Bài>_data.zip
```

- `xay_de.py cv0 cv1 ... nlp7`: dựng từng bài (cv0 là Nordland_PlaceRecognition, cv1 FGVC_Aircraft_Noisy, cv2 Kvasir_PolypSegmentation, cv3 MVTec_AnomalyDetection, cv4 UCF101_ActionRecognition, cv5 VOC_MultiLabel, nlp1 VSFC_Sentiment, nlp2 PhoNER_COVID19, nlp3 Vietnamese_Diacritics, nlp4 IWSLT_EnVi_Translation, nlp5 ViQuAD_PassageRetrieval, nlp6 DimABSA_VA, nlp7 DimABSA_Quad).
- `SELFHOST_RAW=<thư mục>`: dùng dữ liệu gốc ở chỗ khác thay cho `selfhost_kaggle/raw/`; `SELFHOST_NORDLAND=<thư mục>` riêng cho Nordland (chứa `dataset_imageNames/` và `anh640/`).
- Bài Nordland dựng theo đề chính thức tác vụ 2 vòng sơ loại OlpAI'26: 4 đoạn tuyến (train 1.400 hè / 14.000 đông, validation 300 / 3.000, public 450, private 460, kho đông test 9.100), ảnh hè của validation và test bị làm hỏng (`lam_hong_vpr`), private test nằm trong `private_test.zip` mã hoá ZipCrypto, mật khẩu ghi ở `host/mat_khau_private_test.txt`.
- Hai bài DimABSA dựng từ DimABSA2026 (SemEval-2026 Task 3, Track A, eng-restaurant, commit bdc93be), đổi tên trường sang định dạng đề OlpAI'26 (`id`, `text`, `aspects` / `quadruplets`, `valence`, `arousal`). Test 1.000 câu mỗi subtask chia 500 public (`test.jsonl`) và 500 private (`private_test.zip` có mật khẩu). `SELFHOST_DIMABSA=<thư mục>` nếu file jsonl gốc nằm chỗ khác `raw/dimabsa/`. Metric `rmse_va` và `continuous_f1` đã đối chiếu với bộ chấm chính thức `metrics_subtask_1_2_3.py`: trùng tới 12 chữ số.
- `kiem_tra_metric.py`: nộp đáp án phải ra điểm tối đa, nộp thiếu dòng phải báo lỗi; ghi `host/nop_dung_het.csv` để nộp thử trên Kaggle.
- `chay_baseline.sh`: điểm baseline ghi vào `host/min_score.txt`, dùng làm Min_Score.
- `cham.py <Bài> <file.csv>`: chấm tại chỗ khi Kaggle lỗi, in điểm Public, Private, toàn bộ.

## Mỗi bài sau khi dựng

```
<Bài>/
├── upload/              dữ liệu cho thí sinh (vào zip)
├── host/                CHỈ người tổ chức giữ
│   ├── solution.csv     đáp án, cột Usage chia Public/Private
│   ├── metric.py        dán vào Kaggle Metric Notebook
│   ├── nop_dung_het.csv nộp thử phải ra điểm tối đa
│   └── min_score.txt    điểm baseline
└── <Bài>_data.zip       tải lên trang Data của cuộc thi
```

## Bài học khi dựng đề

- Id phải có tiền tố chữ (`im000123`, không phải `000123`): pandas đọc id toàn chữ số thành số và làm mất số 0 ở đầu, file nộp sẽ không khớp đáp án.
- Mask của PASCAL VOC trên Hugging Face lưu dạng màu RGB theo bảng màu VOC, không phải chỉ số lớp.
- UIT-VSFC có vài câu rỗng; TF-IDF sẽ lỗi nếu không bỏ.
- MVTec AD trên Hugging Face gồm hàng nghìn file nhỏ, tải cả repo một lúc sẽ bị lỗi 429; chỉ lấy các loại cần và thử lại khi lỗi.
- Với bài có các mẫu kề nhau gần giống hệt (khung video liền nhau), chia Public/Private theo khối liền nhau, không chia ngẫu nhiên từng dòng.
