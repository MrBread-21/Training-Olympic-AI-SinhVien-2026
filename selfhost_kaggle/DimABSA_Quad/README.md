# Cảm xúc đa chiều theo khía cạnh - Subtask 2: trích bộ tứ (DimABSA)

Dựng lại **Subtask 2 của đề NLP OlpAI'26 "Dimensional Aspect-Based Sentiment Analysis"** (tài liệu AIO2026 gọi là Đề 1): chỉ cho câu, tự tìm mọi bộ tứ (aspect, category, opinion, valence-arousal), chấm bằng cF1. Đề thật dùng tiếng Việt và dữ liệu không công bố; bản này dùng dữ liệu công khai DimABSA2026 (SemEval-2026 Task 3, tiếng Anh, review nhà hàng). Định dạng file và cách chấm giữ như đề, metric khớp bộ chấm chính thức. Subtask 1 (hồi quy VA khi biết khía cạnh) là cuộc thi riêng: [DimABSA_VA](../DimABSA_VA/).

## Overview

Một câu nhận xét thường nói về nhiều khía cạnh với thái độ khác nhau. Bài toán **DimABSA** đặt cảm xúc của mỗi khía cạnh lên mặt phẳng liên tục: **valence** từ tiêu cực (1) đến tích cực (9), **arousal** từ bình thản (1) đến kích động (9).

**Nhiệm vụ**: với mỗi câu, tìm tập bộ tứ, mỗi bộ gồm:

- `aspect`: cụm từ chỉ đối tượng được đánh giá, chép đúng như trong câu; khía cạnh ẩn ghi `"NULL"`.
- `category`: nhãn dạng `ENTITY#ATTRIBUTE`, ví dụ `FOOD#QUALITY`, `SERVICE#GENERAL`.
- `opinion`: cụm từ nêu ý kiến, chép đúng như trong câu; ý kiến ẩn ghi `"NULL"`.
- `valence`, `arousal`: trong [1,00; 9,00], làm tròn 2 chữ số thập phân.

```
// đầu vào (một dòng của test.jsonl)
{"id": "PUB_0007", "text": "great pizza , poor service"}
// đầu ra (một dòng của file dự đoán)
{"id": "PUB_0007", "quadruplets": [
  {"aspect": "pizza", "category": "FOOD#QUALITY", "opinion": "great", "valence": 7.50, "arousal": 7.67},
  {"aspect": "service", "category": "SERVICE#GENERAL", "opinion": "poor", "valence": 3.33, "arousal": 5.83}]}
```

**Luật (giống vòng khu vực OlpAI'26)**:

- Không dùng bất kỳ model ngôn ngữ pretrained nào (BERT, RoBERTa, PhoBERT, Gemma, LLM, word embedding tải sẵn...). Được dùng kiến trúc có sẵn trong thư viện và train từ đầu trên dữ liệu đề.
- Không dùng dữ liệu ngoài bộ được cấp. Không gán nhãn tay cho test.
- 5 tiếng đầu: tối đa 20 lần nộp public. Tiếng cuối: mật khẩu `private_test.zip` được công bố, tối đa 5 lần nộp private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.

## Evaluation

**Continuous F1 (cF1)**, thang 0-1, cao hơn là tốt hơn. Trong mỗi câu:

- Một dự đoán **khớp** nhãn chuẩn khi bộ ba (aspect, category, opinion) trùng hoàn toàn sau khi chuyển chữ thường. Lệch một từ ở ranh giới cụm (`best` so với `one of the best`) là không khớp.
- Khi khớp: `cTP = max(0, 1 - d / sqrt(128))`, với d là khoảng cách Euclid giữa (V, A) dự đoán và nhãn chuẩn; sqrt(128) ≈ 11,3137 là khoảng cách lớn nhất trên [1, 9] x [1, 9]. VA ngoài [1; 9] cho cTP = 0.
- Nhãn chuẩn khớp với **nhiều** dự đoán trùng nhau: nhãn đó nhận cTP = 0 và tính là bỏ sót; các dự đoán trùng đều tính là dự đoán sai.

```
cPrecision = tổng cTP / (TP + FP)      cRecall = tổng cTP / (TP + FN)      cF1 = 2 x cP x cR / (cP + cR)
```

TP là số nhãn chuẩn khớp đúng một dự đoán, FP là số dự đoán không khớp, FN là số nhãn chuẩn bị bỏ sót. Câu không có dòng nộp coi như không dự đoán gì.

Ví dụ: câu "I ordered the Pad Thai and it was bad", nhãn (Pad Thai, FOOD#QUALITY, bad, 4,00; 5,12), dự đoán đúng bộ ba với VA (2,73; 7,12): d ≈ 2,369, cTP ≈ 1 - 2,369 / 11,314 ≈ 0,791. Cùng bộ ba nhưng opinion chỉ là một phần cụm chuẩn thì cTP = 0.

Mốc tham khảo trên test DimABSA tiếng Anh (theo tài liệu AIO, dùng LLM pretrained nên không áp dụng được ở đây): Gemma-3 12B QLoRA + self-consistency đạt cF1 khoảng 0,64.

## Submission File

Đề thật nộp file JSON Lines như ví dụ ở trên. Kaggle chỉ nhận CSV nên mỗi câu là một dòng `id,quadruplets`, cột `quadruplets` là danh sách JSON (danh sách rỗng `[]` nếu không tìm thấy gì):

```
id,quadruplets
PUB_0007,"[{""aspect"": ""pizza"", ""category"": ""FOOD#QUALITY"", ""opinion"": ""great"", ""valence"": 7.5, ""arousal"": 7.67}]"
```

Với pandas: `pd.DataFrame({"id": ids, "quadruplets": [json.dumps(x) for x in ds]}).to_csv("submission.csv", index=False)`. Baseline ghi cả hai dạng (`pred_test.jsonl` và `submission.csv`).

Public là 500 câu trong `test.jsonl`, private là 500 câu trong `private_test.zip`. Nên nộp đủ 1.000 dòng; trước khi có mật khẩu, giữ nguyên các dòng `PRV_...` như `sample_submission.csv`.

## Data

```
train.jsonl          2.284 câu, 3.659 bộ tứ: {"id", "text", "quadruplets": [{"aspect", "category", "opinion", "valence", "arousal"}]}
dev.jsonl            200 câu, 408 bộ tứ, cùng định dạng (dùng để tự chấm)
test.jsonl           500 câu public: {"id", "text"}
private_test.zip     (có mật khẩu) private_test.jsonl: 500 câu private, cùng định dạng test.jsonl
sample_submission.csv  id,quadruplets cho cả 1.000 câu (danh sách rỗng)
baseline.ipynb, baseline.py  từ điển khía cạnh / opinion rút từ train, có tự chấm trên dev
LICENSE-DATASET.txt
```

Điều cần để ý khi làm bài:

- **NULL**: train có 880 aspect NULL và 699 opinion NULL, còn dev và test không có cái nào. Dự đoán NULL trên test chắc chắn sai. (Đề thật tiếng Việt thì cho phép NULL ở test.)
- **Câu test dày hơn**: trung bình 1,60 bộ tứ mỗi câu ở train, khoảng 2,1 ở dev và test; 61% câu train chỉ có 1 bộ tứ.
- **Category lệch**: `FOOD#QUALITY`, `SERVICE#GENERAL`, `RESTAURANT#GENERAL` chiếm khoảng 69% train; `AMBIENCE#STYLE_OPTIONS` không có trong train nhưng có trong test.
- **Định dạng chữ**: câu train đã tách từ và viết thường (`ca n ' t wait`), dev và test là câu gốc. Aspect và opinion trong nhãn luôn là chuỗi con của câu.
- cF1 phạt trọn điểm khi lệch ranh giới cụm từ, còn sai VA chỉ bị trừ một phần: ưu tiên trích đúng chuỗi và category trước, tinh chỉnh VA sau.

Giải nén private test (khi có mật khẩu):

```python
import zipfile
zipfile.ZipFile("private_test.zip").extractall(".", pwd=b"<mật khẩu>")   # hoặc: unzip -P <mật khẩu> private_test.zip
```

## Khác biệt với đề thật

| | Đề thật | Bản này |
|---|---|---|
| Ngôn ngữ, miền | Tiếng Việt có teencode, lỗi chính tả, chêm tiếng Anh; ứng dụng, giao hàng, món ăn | Tiếng Anh, review nhà hàng |
| Category | 10 nhãn: `APP#PERFORMANCE`, `APP#UI_UX`, `STAFF#ATTITUDE`, `STAFF#PROFESSIONALISM`, `PRODUCT#QUALITY`, `PRODUCT#DESIGN`, `SERVICE#TIME`, `SERVICE#GENERAL`, `PRICE#RATES`, `PRICE#PROMO` | Nhãn miền nhà hàng (`FOOD#QUALITY`, `SERVICE#GENERAL`, `AMBIENCE#GENERAL`...) |
| NULL ở test | Có thể có | Không có |
| Model pretrained | Tài liệu AIO dùng Gemma-3 QLoRA | Cấm theo luật vòng khu vực: chỉ kiến trúc, train từ đầu |
| File nộp | JSON Lines | CSV `id,quadruplets`, cột quadruplets là JSON |
| Nộp private | Nộp riêng ở tiếng cuối | Kaggle chấm cả 1.000 dòng trong một file, cột `Usage` tách public/private |

Nguồn: SemEval-2026 Task 3 DimABSA (github.com/DimABSA/DimABSA2026, Track A, eng-restaurant, subtask 3 DimASQP). Kho gốc không ghi giấy phép, chỉ dùng luyện tập nội bộ. Tài liệu đề: AIO2026 "Dimensional Aspect-Based Sentiment Analysis (Olympic AI)" và repo Purin1410/Olympic_AI_MET_2026_preliminary_round.
