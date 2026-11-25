# Cảm xúc đa chiều theo khía cạnh - Subtask 1: hồi quy Valence-Arousal (DimABSA)

Dựng lại **Subtask 1 của đề NLP OlpAI'26 "Dimensional Aspect-Based Sentiment Analysis"** (tài liệu AIO2026 gọi là Đề 1): cho câu và các khía cạnh, dự đoán cặp tọa độ cảm xúc (valence, arousal) cho từng khía cạnh, chấm bằng RMSE_VA. Đề thật dùng tiếng Việt và dữ liệu không công bố; bản này dùng dữ liệu công khai DimABSA2026 (SemEval-2026 Task 3, tiếng Anh, review nhà hàng). Định dạng file, thang điểm và cách chấm giữ như đề. Subtask 2 (trích bộ tứ) là cuộc thi riêng: [DimABSA_Quad](../DimABSA_Quad/).

## Overview

Một câu nhận xét thường nói về nhiều khía cạnh với thái độ khác nhau: khen phục vụ nhưng chê món ăn. Thay vì gán nhãn rời rạc tích cực / tiêu cực / trung tính, bài toán **DimABSA** đặt cảm xúc của mỗi khía cạnh lên mặt phẳng liên tục theo mô hình vòng tròn của Russell:

- **Valence** (V): từ tiêu cực (1) đến tích cực (9).
- **Arousal** (A): từ bình thản (1) đến kích động (9). Ví dụ người giận dữ có V dưới 3 và A trên 7.

**Nhiệm vụ**: với mỗi câu và danh sách khía cạnh cho trước, dự đoán (valence, arousal) cho từng khía cạnh. Giá trị nằm trong [1,00; 9,00], làm tròn 2 chữ số thập phân.

```
// đầu vào (một dòng của test.jsonl)
{"id": "PUB_0002", "text": "... offers such lousy service", "aspects": [{"term": "location"}, {"term": "service"}]}
// đầu ra (một dòng của file dự đoán)
{"id": "PUB_0002", "aspects": [{"aspect": "location", "valence": 7.10, "arousal": 6.80},
                               {"aspect": "service", "valence": 2.30, "arousal": 6.90}]}
```

**Luật (giống vòng khu vực OlpAI'26)**:

- Không dùng bất kỳ model ngôn ngữ pretrained nào (BERT, RoBERTa, PhoBERT, LLM, word embedding tải sẵn...). Được dùng kiến trúc có sẵn trong thư viện (ví dụ `transformers` khởi tạo ngẫu nhiên) và train từ đầu trên dữ liệu đề.
- Không dùng dữ liệu ngoài bộ được cấp. Không gán nhãn tay cho test.
- 5 tiếng đầu: tối đa 20 lần nộp public. Tiếng cuối: mật khẩu `private_test.zip` được công bố, tối đa 5 lần nộp private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.

## Evaluation

**RMSE_VA** (thấp hơn là tốt hơn):

```
RMSE_VA = sqrt( 1/N x tổng_i [ (Vp_i - Vg_i)^2 + (Ap_i - Ag_i)^2 ] )
```

N là tổng số cặp (câu, khía cạnh) của tập đang chấm; (Vp, Ap) là dự đoán, (Vg, Ag) là nhãn chuẩn.

- Dự đoán được ghép với khía cạnh theo đúng chuỗi `aspect` (giống bộ chấm chính thức), nên hãy chép nguyên chuỗi `term` của đầu vào.
- Khía cạnh không có dự đoán, giá trị không đọc được hoặc nằm ngoài [1; 9] bị tính sai số lớn nhất có thể: trên mỗi chiều lấy khoảng cách từ nhãn chuẩn tới biên 1 hoặc 9, biên nào xa hơn.
- Ví dụ: hai khía cạnh, nhãn (7,50; 7,67) và (3,33; 5,83), dự đoán (7,00; 7,00) và (4,00; 6,00). Tổng bình phương sai số = 0,25 + 0,4489 + 0,4489 + 0,0289 = 1,1767, RMSE_VA = sqrt(1,1767 / 2) = 0,7670.
- Mốc tham khảo trên test của DimABSA tiếng Anh (theo tài liệu AIO, dùng model pretrained nên không áp dụng được ở đây): đoán trung bình 2,197; LogSigma (RoBERTa-large, 3 seed) 1,1045.

## Submission File

Đề thật nộp file JSON Lines như ví dụ ở trên. Kaggle chỉ nhận CSV nên mỗi câu là một dòng `id,aspects`, cột `aspects` là danh sách JSON của dòng dự đoán đó:

```
id,aspects
PUB_0002,"[{""aspect"": ""location"", ""valence"": 7.1, ""arousal"": 6.8}, {""aspect"": ""service"", ""valence"": 2.3, ""arousal"": 6.9}]"
```

Với pandas: `pd.DataFrame({"id": ids, "aspects": [json.dumps(x) for x in ds]}).to_csv("submission.csv", index=False)`. Baseline ghi cả hai dạng (`pred_test.jsonl` và `submission.csv`).

Public là 500 câu trong `test.jsonl`, private là 500 câu trong `private_test.zip`. Nên nộp đủ 1.000 dòng; trước khi có mật khẩu, giữ nguyên các dòng `PRV_...` như `sample_submission.csv`.

## Data

```
train.jsonl          1.648 câu, 2.779 cặp khía cạnh có nhãn: {"id", "text", "aspects": [{"aspect", "valence", "arousal"}]}
dev.jsonl            200 câu, 340 cặp, cùng định dạng (dùng để tự chấm)
test.jsonl           500 câu public: {"id", "text", "aspects": [{"term"}]}
private_test.zip     (có mật khẩu) private_test.jsonl: 500 câu private, cùng định dạng test.jsonl
sample_submission.csv  id,aspects cho cả 1.000 câu (đoán 5,0 / 5,0)
baseline.ipynb, baseline.py  TF-IDF cửa sổ quanh khía cạnh + Ridge, có tự chấm trên dev
LICENSE-DATASET.txt
```

- Train lấy từ file train của DimABSA (bỏ khía cạnh ẩn NULL). Một câu có thể lặp lại cùng một khía cạnh với nhãn khác nhau (khía cạnh được nhắc với nhiều ý kiến).
- Câu train đã tách từ và viết thường (`ca n ' t wait`), còn dev và test là câu gốc có chữ hoa và dấu câu dính liền. Đây là đặc điểm của dữ liệu gốc, cần xử lý khi tiền xử lý.
- Phân bố: valence lệch về tích cực (đỉnh ở 7-8), arousal hẹp (độ lệch chuẩn khoảng 1,1, ít khi dưới 4) và tăng khi valence xa mức 5.

Giải nén private test (khi có mật khẩu):

```python
import zipfile
zipfile.ZipFile("private_test.zip").extractall(".", pwd=b"<mật khẩu>")   # hoặc: unzip -P <mật khẩu> private_test.zip
```

## Khác biệt với đề thật

| | Đề thật | Bản này |
|---|---|---|
| Ngôn ngữ, miền | Tiếng Việt có teencode, lỗi chính tả, chêm tiếng Anh; ứng dụng, giao hàng, món ăn | Tiếng Anh, review nhà hàng |
| Kích thước | Không công bố | Train 1.648 câu, dev 200, test 1.000 (từ DimABSA eng-restaurant) |
| Model pretrained | Tài liệu AIO dùng RoBERTa, PhoBERT | Cấm theo luật vòng khu vực: chỉ kiến trúc, train từ đầu |
| File nộp | JSON Lines | CSV `id,aspects`, cột aspects là JSON |
| Nộp private | Nộp riêng ở tiếng cuối | Kaggle chấm cả 1.000 dòng trong một file, cột `Usage` tách public/private |

Nguồn: SemEval-2026 Task 3 DimABSA (github.com/DimABSA/DimABSA2026, Track A, eng-restaurant). Kho gốc không ghi giấy phép, chỉ dùng luyện tập nội bộ. Tài liệu đề: AIO2026 "Dimensional Aspect-Based Sentiment Analysis (Olympic AI)" và repo Purin1410/Olympic_AI_MET_2026_preliminary_round.
