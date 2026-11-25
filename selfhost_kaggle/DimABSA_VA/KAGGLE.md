# DimABSA_VA - nội dung từng ô trên Kaggle

## Title

OlpAI thử - DimABSA Subtask 1: Valence-Arousal

## Competition URL

pka-olpai-test-dimabsa-va

## Subtitle

Dự đoán tọa độ cảm xúc (valence, arousal) cho từng khía cạnh cho trước trong câu review (Subtask 1 đề NLP OlpAI'26).

## Overview

Dựng lại Subtask 1 đề NLP OlpAI'26 "Dimensional Aspect-Based Sentiment Analysis": cho câu và các khía cạnh, dự đoán valence (tiêu cực 1 - tích cực 9) và arousal (bình thản 1 - kích động 9) cho từng khía cạnh. Chấm bằng RMSE_VA, thấp hơn là tốt hơn. Không dùng model ngôn ngữ pretrained.

## Description

### Bối cảnh

Một câu nhận xét thường nói về nhiều khía cạnh với thái độ khác nhau: khen phục vụ nhưng chê món ăn. Thay vì gán nhãn rời rạc tích cực / tiêu cực / trung tính, bài toán DimABSA đặt cảm xúc của mỗi khía cạnh lên mặt phẳng liên tục theo mô hình vòng tròn của Russell: **valence** đo chiều hướng từ tiêu cực (1) đến tích cực (9), **arousal** đo cường độ từ bình thản (1) đến kích động (9). Ví dụ người giận dữ có valence dưới 3 và arousal trên 7.

### Nhiệm vụ

Với mỗi câu trong `test.jsonl` và danh sách khía cạnh cho trước (`aspects[].term`), dự đoán (valence, arousal) cho từng khía cạnh, trong [1,00; 9,00], làm tròn 2 chữ số thập phân.

```
đầu vào:  {"id": "PUB_0002", "text": "... offers such lousy service", "aspects": [{"term": "location"}, {"term": "service"}]}
đầu ra:   {"id": "PUB_0002", "aspects": [{"aspect": "location", "valence": 7.10, "arousal": 6.80}, {"aspect": "service", "valence": 2.30, "arousal": 6.90}]}
```

### Khác biệt với đề thật

Đề thật là tiếng Việt (có teencode, lỗi chính tả, chêm tiếng Anh) và dữ liệu không công bố. Bản này dùng dữ liệu công khai DimABSA2026 (SemEval-2026 Task 3, tiếng Anh, review nhà hàng), đổi tên trường sang đúng định dạng đề. Đề thật nộp JSON Lines; Kaggle chỉ nhận CSV nên cột `aspects` chứa danh sách JSON.

## Evaluation

**RMSE_VA** (thấp hơn là tốt hơn):

```
RMSE_VA = sqrt( 1/N x tổng_i [ (Vp_i - Vg_i)^2 + (Ap_i - Ag_i)^2 ] )
```

N là tổng số cặp (câu, khía cạnh) của tập đang chấm.

- Dự đoán được ghép với khía cạnh theo đúng chuỗi `aspect`, hãy chép nguyên chuỗi `term` của đầu vào.
- Khía cạnh không có dự đoán, giá trị không đọc được hoặc nằm ngoài [1; 9] bị tính sai số lớn nhất có thể (trên mỗi chiều, khoảng cách từ nhãn chuẩn tới biên 1 hoặc 9, biên nào xa hơn).
- File thiếu cột `id` hoặc `aspects` bị báo lỗi.

Ví dụ: nhãn (7,50; 7,67) và (3,33; 5,83), dự đoán (7,00; 7,00) và (4,00; 6,00): RMSE_VA = sqrt((0,25 + 0,4489 + 0,4489 + 0,0289) / 2) = 0,7670.

## Submission File

Mỗi câu một dòng `id,aspects`, cột `aspects` là danh sách JSON:

```
id,aspects
PUB_0002,"[{""aspect"": ""location"", ""valence"": 7.1, ""arousal"": 6.8}, {""aspect"": ""service"", ""valence"": 2.3, ""arousal"": 6.9}]"
```

Với pandas: `pd.DataFrame({"id": ids, "aspects": [json.dumps(x) for x in ds]}).to_csv("submission.csv", index=False)`.

Public là 500 câu trong `test.jsonl`, private là 500 câu trong `private_test.zip`. Nên nộp đủ 1.000 dòng; trước khi có mật khẩu, giữ nguyên các dòng `PRV_...` như trong `sample_submission.csv`.

## Timeline

- Bắt đầu: do người tổ chức thông báo.
- Giờ 0-5: làm bài, tối đa 20 lần nộp, chỉ có public test.
- Giờ thứ 6: công bố mật khẩu `private_test.zip`; dự đoán private, viết `main.py`, chọn bài nộp cuối.
- Kết thúc: 6 tiếng sau giờ bắt đầu; bảng private được mở ngay sau đó.

## Rules

- Không dùng bất kỳ model ngôn ngữ pretrained nào (BERT, RoBERTa, PhoBERT, LLM, word embedding tải sẵn...). Được dùng kiến trúc có sẵn trong thư viện, khởi tạo ngẫu nhiên, train từ đầu trên dữ liệu đề.
- Không dùng dữ liệu ngoài bộ được cấp. Không gán nhãn tay cho test.
- Không mở `private_test.zip` trước khi mật khẩu được công bố.
- Tối đa 20 lần nộp; chọn bài nộp cuối để chấm private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.

## Citation

SemEval-2026 Task 3: Dimensional Aspect-Based Sentiment Analysis (DimABSA), github.com/DimABSA/DimABSA2026 (Track A, eng-restaurant). Tài liệu đề: AIO2026 "Dimensional Aspect-Based Sentiment Analysis (Olympic AI)". Đề gốc: Olympic Trí tuệ nhân tạo sinh viên Việt Nam 2026.

## Dataset Description

Dữ liệu dạng JSON Lines, mỗi dòng một câu. Câu train đã tách từ và viết thường (`ca n ' t wait`), dev và test là câu gốc.

### Files

- `train.jsonl`: 1.648 câu, 2.779 cặp khía cạnh có nhãn, dạng `{"id", "text", "aspects": [{"aspect", "valence", "arousal"}]}`. Một câu có thể lặp lại cùng khía cạnh với nhãn khác nhau.
- `dev.jsonl`: 200 câu, 340 cặp, cùng định dạng, dùng để tự chấm.
- `test.jsonl`: 500 câu public, dạng `{"id", "text", "aspects": [{"term"}]}`.
- `private_test.zip` (có mật khẩu): `private_test.jsonl`, 500 câu private. Giải nén: `zipfile.ZipFile("private_test.zip").extractall(".", pwd=b"<mật khẩu>")`.
- `sample_submission.csv`: `id,aspects` cho cả 1.000 câu (đoán 5,0 / 5,0).
- `baseline.ipynb` / `baseline.py`: TF-IDF cửa sổ quanh khía cạnh + Ridge, có tự chấm trên dev; ghi `pred_test.jsonl` (định dạng đề) và `submission.csv`.

## Data license

Other (specified in description): dữ liệu DimABSA2026 không ghi giấy phép, chỉ dùng luyện tập nội bộ, không phát tán lại.

## Files để tải lên

- Data: `DimABSA_VA_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py` (hàm `rmse_va`), chọn **thấp hơn là tốt hơn** (sort ascending)
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Mật khẩu private test: `host/mat_khau_private_test.txt`, công bố ở tiếng cuối
- Kiểm tra bằng sandbox submission: nộp `host/nop_dung_het.csv` phải ra 0, nộp `submission.csv` của baseline phải gần `host/min_score.txt` (public 1,9014)
