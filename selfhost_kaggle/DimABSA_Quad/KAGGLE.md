# DimABSA_Quad - nội dung từng ô trên Kaggle

## Title

OlpAI thử - DimABSA Subtask 2: trích bộ tứ

## Competition URL

pka-olpai-test-dimabsa-quad

## Subtitle

Từ câu review, trích mọi bộ tứ (aspect, category, opinion, valence-arousal) (Subtask 2 đề NLP OlpAI'26).

## Overview

Dựng lại Subtask 2 đề NLP OlpAI'26 "Dimensional Aspect-Based Sentiment Analysis": chỉ cho câu, tìm mọi khía cạnh, nhãn danh mục, cụm ý kiến và tọa độ cảm xúc (valence, arousal) của từng khía cạnh. Chấm bằng Continuous F1 (cF1). Không dùng model ngôn ngữ pretrained.

## Description

### Bối cảnh

Một câu nhận xét thường nói về nhiều khía cạnh với thái độ khác nhau. Bài toán DimABSA đặt cảm xúc của mỗi khía cạnh lên mặt phẳng liên tục: **valence** từ tiêu cực (1) đến tích cực (9), **arousal** từ bình thản (1) đến kích động (9).

### Nhiệm vụ

Với mỗi câu trong `test.jsonl`, tìm tập bộ tứ, mỗi bộ gồm:

- `aspect`: cụm từ chỉ đối tượng được đánh giá, chép đúng như trong câu (khía cạnh ẩn ghi `"NULL"`);
- `category`: nhãn `ENTITY#ATTRIBUTE`, ví dụ `FOOD#QUALITY`, `SERVICE#GENERAL`;
- `opinion`: cụm từ nêu ý kiến, chép đúng như trong câu (ý kiến ẩn ghi `"NULL"`);
- `valence`, `arousal`: trong [1,00; 9,00], làm tròn 2 chữ số thập phân.

```
đầu vào:  {"id": "PUB_0007", "text": "great pizza , poor service"}
đầu ra:   {"id": "PUB_0007", "quadruplets": [
             {"aspect": "pizza", "category": "FOOD#QUALITY", "opinion": "great", "valence": 7.50, "arousal": 7.67},
             {"aspect": "service", "category": "SERVICE#GENERAL", "opinion": "poor", "valence": 3.33, "arousal": 5.83}]}
```

### Khác biệt với đề thật

Đề thật là tiếng Việt, miền ứng dụng / giao hàng / món ăn, 10 nhãn category (`APP#PERFORMANCE`, `APP#UI_UX`, `STAFF#ATTITUDE`, `STAFF#PROFESSIONALISM`, `PRODUCT#QUALITY`, `PRODUCT#DESIGN`, `SERVICE#TIME`, `SERVICE#GENERAL`, `PRICE#RATES`, `PRICE#PROMO`) và cho phép NULL ở test. Bản này dùng dữ liệu công khai DimABSA2026 (SemEval-2026 Task 3, tiếng Anh, review nhà hàng): category miền nhà hàng, test không có NULL. Đề thật nộp JSON Lines; Kaggle chỉ nhận CSV nên cột `quadruplets` chứa danh sách JSON.

## Evaluation

**Continuous F1 (cF1)**, thang 0-1, khớp bộ chấm chính thức của DimABSA. Trong mỗi câu:

- Dự đoán **khớp** nhãn chuẩn khi (aspect, category, opinion) trùng hoàn toàn sau khi chuyển chữ thường. Lệch một từ ở ranh giới cụm là không khớp.
- Khi khớp: `cTP = max(0, 1 - d / sqrt(128))`, d là khoảng cách Euclid giữa (V, A) dự đoán và nhãn chuẩn. VA ngoài [1; 9] cho cTP = 0.
- Nhãn chuẩn khớp với nhiều dự đoán trùng nhau: nhãn đó nhận cTP = 0 và tính là bỏ sót; các dự đoán trùng đều tính là dự đoán sai.

```
cPrecision = tổng cTP / (TP + FP)      cRecall = tổng cTP / (TP + FN)      cF1 = 2 x cP x cR / (cP + cR)
```

Câu không có dòng nộp coi như không dự đoán gì. File thiếu cột `id` hoặc `quadruplets` bị báo lỗi.

Ví dụ: nhãn (Pad Thai, FOOD#QUALITY, bad, 4,00; 5,12), dự đoán đúng bộ ba với VA (2,73; 7,12): d ≈ 2,369, cTP ≈ 0,791. Opinion chỉ là một phần cụm chuẩn (`best` so với `one of the best`) thì cTP = 0.

## Submission File

Mỗi câu một dòng `id,quadruplets`, cột `quadruplets` là danh sách JSON (`[]` nếu không tìm thấy gì):

```
id,quadruplets
PUB_0007,"[{""aspect"": ""pizza"", ""category"": ""FOOD#QUALITY"", ""opinion"": ""great"", ""valence"": 7.5, ""arousal"": 7.67}]"
```

Với pandas: `pd.DataFrame({"id": ids, "quadruplets": [json.dumps(x) for x in ds]}).to_csv("submission.csv", index=False)`.

Public là 500 câu trong `test.jsonl`, private là 500 câu trong `private_test.zip`. Nên nộp đủ 1.000 dòng; trước khi có mật khẩu, giữ nguyên các dòng `PRV_...` như trong `sample_submission.csv`.

## Timeline

- Bắt đầu: do người tổ chức thông báo.
- Giờ 0-5: làm bài, tối đa 20 lần nộp, chỉ có public test.
- Giờ thứ 6: công bố mật khẩu `private_test.zip`; dự đoán private, viết `main.py`, chọn bài nộp cuối.
- Kết thúc: 6 tiếng sau giờ bắt đầu; bảng private được mở ngay sau đó.

## Rules

- Không dùng bất kỳ model ngôn ngữ pretrained nào (BERT, RoBERTa, PhoBERT, Gemma, LLM, word embedding tải sẵn...). Được dùng kiến trúc có sẵn trong thư viện, khởi tạo ngẫu nhiên, train từ đầu trên dữ liệu đề.
- Không dùng dữ liệu ngoài bộ được cấp. Không gán nhãn tay cho test.
- Không mở `private_test.zip` trước khi mật khẩu được công bố.
- Tối đa 20 lần nộp; chọn bài nộp cuối để chấm private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.

## Citation

SemEval-2026 Task 3: Dimensional Aspect-Based Sentiment Analysis (DimABSA), github.com/DimABSA/DimABSA2026 (Track A, eng-restaurant, DimASQP). Tài liệu đề: AIO2026 "Dimensional Aspect-Based Sentiment Analysis (Olympic AI)". Đề gốc: Olympic Trí tuệ nhân tạo sinh viên Việt Nam 2026.

## Dataset Description

Dữ liệu dạng JSON Lines, mỗi dòng một câu. Câu train đã tách từ và viết thường (`ca n ' t wait`), dev và test là câu gốc. Train có 880 aspect NULL và 699 opinion NULL, dev và test không có NULL. Trung bình 1,60 bộ tứ mỗi câu ở train, khoảng 2,1 ở dev và test.

### Files

- `train.jsonl`: 2.284 câu, 3.659 bộ tứ, dạng `{"id", "text", "quadruplets": [{"aspect", "category", "opinion", "valence", "arousal"}]}`.
- `dev.jsonl`: 200 câu, 408 bộ tứ, cùng định dạng, dùng để tự chấm.
- `test.jsonl`: 500 câu public, dạng `{"id", "text"}`.
- `private_test.zip` (có mật khẩu): `private_test.jsonl`, 500 câu private. Giải nén: `zipfile.ZipFile("private_test.zip").extractall(".", pwd=b"<mật khẩu>")`.
- `sample_submission.csv`: `id,quadruplets` cho cả 1.000 câu (danh sách rỗng).
- `baseline.ipynb` / `baseline.py`: từ điển khía cạnh và opinion rút từ train, có tự chấm cF1 trên dev; ghi `pred_test.jsonl` (định dạng đề) và `submission.csv`.

## Data license

Other (specified in description): dữ liệu DimABSA2026 không ghi giấy phép, chỉ dùng luyện tập nội bộ, không phát tán lại.

## Files để tải lên

- Data: `DimABSA_Quad_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py` (hàm `continuous_f1`), chọn cao hơn là tốt hơn
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Mật khẩu private test: `host/mat_khau_private_test.txt`, công bố ở tiếng cuối
- Kiểm tra bằng sandbox submission: nộp `host/nop_dung_het.csv` phải ra 1, nộp `submission.csv` của baseline phải gần `host/min_score.txt` (public 0,3053)
