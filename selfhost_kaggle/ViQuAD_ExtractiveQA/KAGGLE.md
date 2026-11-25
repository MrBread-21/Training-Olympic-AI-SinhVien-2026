# ViQuAD_ExtractiveQA - nội dung từng ô trên Kaggle

## Title

OlpAI thử - Hỏi đáp trích xuất tiếng Việt

## Competition URL

pka-olpai-test-viquad-qa

## Subtitle

Chép câu trả lời từ đoạn văn Wikipedia tiếng Việt, hoặc để trống nếu không có (dạng bài VIQA).

## Overview

Cho đoạn văn và câu hỏi, trích cụm từ trả lời hoặc để trống khi đoạn văn không có câu trả lời (khoảng 30% câu hỏi). Chấm bằng F1 theo từ. Không dùng model ngôn ngữ pretrained.

## Description

Dạng bài **hỏi đáp đọc hiểu**: cho đoạn văn và câu hỏi, chép ra cụm từ trong đoạn văn trả lời câu hỏi, hoặc để trống nếu đoạn văn không chứa câu trả lời. Đây là dạng bài VIQA đội đã luyện, và là dạng "trích span" hay gặp trong đề NLP không dùng pretrained.

Mỗi mẫu gồm một đoạn văn Wikipedia tiếng Việt và một câu hỏi về đoạn văn đó. Khoảng **một phần ba câu hỏi không có câu trả lời** trong đoạn văn (câu hỏi được viết để trông giống câu hỏi trả lời được), khi đó phải để trống.

**Nhiệm vụ**: với mỗi câu hỏi trong test, trả về chuỗi trả lời chép nguyên văn từ `context`, hoặc chuỗi rỗng nếu không có câu trả lời.

## Evaluation

**F1 theo từ** (kiểu SQuAD 2.0), thang 0-1, trung bình trên mọi câu hỏi:

- Chuẩn hoá cả câu trả lời và đáp án: Unicode NFC, chữ thường, bỏ dấu câu, tách theo khoảng trắng.
- Câu hỏi có đáp án: `F1 = 2 x P x R / (P + R)`, với P = số từ trùng / số từ của câu trả lời, R = số từ trùng / số từ của đáp án (đếm trùng có lặp). Có nhiều đáp án đúng thì lấy F1 cao nhất. Để trống thì được 0.
- Câu hỏi không có đáp án: để trống được 1, trả lời bất cứ gì được 0.

Ví dụ: đáp án "Lâm Bá Kiệt", trả lời "tên Lâm Bá Kiệt": P = 3/4, R = 3/3, F1 = 0,857.

Lưu ý: vì khoảng 30% câu hỏi không có đáp án, nộp toàn chuỗi rỗng đã được khoảng 0,27-0,33. Baseline chỉ nhỉnh hơn mốc này; phải trích đúng cụm ngắn mới tăng điểm đáng kể.

## Submission File

`submission.csv` có tiêu đề `id,answer`, mỗi câu hỏi một dòng; `answer` rỗng nghĩa là không có đáp án:

```
id,answer
q013579,Lâm Bá Kiệt
q024680,
```

Public là các câu hỏi trong `test.csv`, private là các câu hỏi trong `private_test.zip`. Nên nộp đủ mọi dòng của cả hai; trước khi có mật khẩu, giữ nguyên các dòng private như `sample_submission.csv`.

## Timeline

- Bắt đầu: do người tổ chức thông báo.
- Giờ 0-5: làm bài, tối đa 20 lần nộp, chỉ có public test.
- Giờ thứ 6: công bố mật khẩu `private_test.zip`; dự đoán private, viết `main.py`, chọn bài nộp cuối.
- Kết thúc: 6 tiếng sau giờ bắt đầu; bảng private được mở ngay sau đó.

## Rules

- Không dùng bất kỳ model ngôn ngữ pretrained nào (PhoBERT, XLM-R, LLM, word embedding tải sẵn...). Được dùng kiến trúc có sẵn trong thư viện, khởi tạo ngẫu nhiên, train từ đầu trên dữ liệu đề.
- Không dùng dữ liệu ngoài bộ được cấp (kể cả bản ViQuAD gốc). Không gán nhãn tay cho test.
- 5 tiếng đầu: tối đa 20 lần nộp public. Tiếng cuối: mật khẩu `private_test.zip` được công bố, tối đa 5 lần nộp private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Không mở `private_test.zip` trước khi mật khẩu được công bố.
- Tối đa 20 lần nộp; chọn bài nộp cuối để chấm private.

## Citation

Nguyen và cộng sự, VLSP 2021 - ViQuAD 2.0: Vietnamese Machine Reading Comprehension. Bản Hugging Face taidng/UIT-ViQuAD2.0.

## Dataset Description

```
train.csv              25.841 câu hỏi trên 126 bài Wikipedia: id, title, context, question, answer, answer_start
dev.csv                2.613 câu hỏi trên 12 bài khác, cùng định dạng (dùng để tự chấm)
test.csv               1.965 câu hỏi public: id, title, context, question
private_test.zip       (có mật khẩu) private_test.csv: 1.849 câu hỏi private, cùng định dạng test.csv
sample_submission.csv  id,answer cho mọi câu hỏi test (để trống)
baseline.ipynb, baseline.py
LICENSE-DATASET.txt
```

- `answer` rỗng và `answer_start = -1` khi câu hỏi không có đáp án; ngược lại `context[answer_start : answer_start + len(answer)] == answer`.
- Train, dev và test là các **bài Wikipedia khác nhau** (test lấy từ 19 bài, public và private chia theo bài): không có đoạn văn nào của test xuất hiện trong train, nên không học thuộc được đáp án theo đoạn văn.
- Một đoạn văn có nhiều câu hỏi.

Giải nén private test (khi có mật khẩu):

```python
import zipfile
zipfile.ZipFile("private_test.zip").extractall(".", pwd=b"<mật khẩu>")   # hoặc: unzip -P <mật khẩu> private_test.zip
```

## Data license

Other (specified in description): UIT-ViQuAD 2.0, chỉ dùng cho học tập và nghiên cứu.

## Files để tải lên

- Data: `ViQuAD_ExtractiveQA_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py` (hàm `qa_f1`), cao hơn là tốt hơn
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Mật khẩu private test: `host/mat_khau_private_test.txt`, công bố ở tiếng cuối
- Kiểm tra bằng sandbox submission: nộp `host/nop_dung_het.csv` (nếu có) phải ra điểm tối đa; nộp `submission.csv` của baseline phải gần `host/min_score.txt` (public 0.2762, private 0.3373)
