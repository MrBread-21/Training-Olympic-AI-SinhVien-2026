# Hỏi đáp trích xuất tiếng Việt (ViQuAD 2.0)

Dạng bài **hỏi đáp đọc hiểu**: cho đoạn văn và câu hỏi, chép ra cụm từ trong đoạn văn trả lời câu hỏi, hoặc để trống nếu đoạn văn không chứa câu trả lời. Đây là dạng bài VIQA đội đã luyện, và là dạng "trích span" hay gặp trong đề NLP không dùng pretrained.

## Overview

Mỗi mẫu gồm một đoạn văn Wikipedia tiếng Việt và một câu hỏi về đoạn văn đó. Khoảng **một phần ba câu hỏi không có câu trả lời** trong đoạn văn (câu hỏi được viết để trông giống câu hỏi trả lời được), khi đó phải để trống.

**Nhiệm vụ**: với mỗi câu hỏi trong test, trả về chuỗi trả lời chép nguyên văn từ `context`, hoặc chuỗi rỗng nếu không có câu trả lời.

**Luật (giống vòng khu vực OlpAI'26)**:

- Không dùng bất kỳ model ngôn ngữ pretrained nào (PhoBERT, XLM-R, LLM, word embedding tải sẵn...). Được dùng kiến trúc có sẵn trong thư viện, khởi tạo ngẫu nhiên, train từ đầu trên dữ liệu đề.
- Không dùng dữ liệu ngoài bộ được cấp (kể cả bản ViQuAD gốc). Không gán nhãn tay cho test.
- 5 tiếng đầu: tối đa 20 lần nộp public. Tiếng cuối: mật khẩu `private_test.zip` được công bố, tối đa 5 lần nộp private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.

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

## Data

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

## Hướng làm gợi ý

- Sinh span ứng viên (cụm 1-10 từ trong câu chứa nhiều từ của câu hỏi), tạo đặc trưng (khoảng cách tới từ của câu hỏi, loại câu hỏi "ai / khi nào / bao nhiêu", chữ số, chữ hoa, độ dài), rồi xếp hạng bằng `XGBRanker` / LightGBM nhóm theo câu hỏi.
- Hoặc BiLSTM / Transformer nhỏ train từ đầu dự đoán vị trí đầu và cuối, thêm một lớp "không có đáp án".
- Chọn ngưỡng "để trống" trên dev: đây là nút vặn ảnh hưởng điểm nhiều nhất.

Nguồn: UIT-ViQuAD 2.0 (VLSP 2021), bản Hugging Face `taidng/UIT-ViQuAD2.0`; test lấy từ tập validation (tập test gốc không có đáp án).
