# SpeechCommands_KWS - nội dung từng ô trên Kaggle

## Title

OlpAI thử - Nhận dạng từ khoá trong âm thanh

## Competition URL

pka-olpai-test-speech-kws

## Subtitle

Nhận dạng 10 lệnh nói và từ lạ trong đoạn âm thanh 1 giây, test là người nói mới và có tạp âm.

## Overview

Phân loại đoạn ghi âm 1 giây thành 10 lệnh (yes, no, up, down, left, right, on, off, stop, go) hoặc unknown. Test là người nói chưa gặp, 70% có tạp âm. Chấm bằng macro-F1.

## Description

Dạng bài **âm thanh**: nghe đoạn ghi âm 1 giây và cho biết người nói đang nói lệnh nào trong 10 lệnh điều khiển, hay một từ khác. Test là **người nói chưa gặp** khi train và phần lớn bị **lẫn tạp âm**.

Mỗi file là một đoạn `.wav` 16 kHz, 16 bit, một kênh, dài tối đa 1 giây. Có 11 lớp:

- 10 lệnh: `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`;
- `unknown`: một trong 25 từ khác (`bed`, `bird`, `cat`, `happy`, `zero`... `nine`, `forward`, `backward`...), chiếm khoảng một nửa dữ liệu.

**Nhiệm vụ**: với mỗi đoạn âm thanh test, dự đoán một trong 11 nhãn.

## Evaluation

**Macro-F1** trên 11 lớp: F1 của từng lớp rồi lấy trung bình không trọng số. Lớp `unknown` chỉ nặng bằng một lệnh, nên đoán `unknown` cho mọi thứ chỉ được khoảng 0,06.

## Submission File

`submission.csv` có tiêu đề `id,label`, mỗi đoạn âm thanh test một dòng:

```
id,label
te012345,stop
te067890,unknown
```

Public là 2.281 file trong `test/`, private là 2.109 file trong `private_test.zip`. Nên nộp đủ 4.390 dòng; trước khi có mật khẩu, giữ nguyên các dòng private như `sample_submission.csv`.

## Timeline

- Bắt đầu: do người tổ chức thông báo.
- Giờ 0-5: làm bài, tối đa 20 lần nộp, chỉ có public test.
- Giờ thứ 6: công bố mật khẩu `private_test.zip`; dự đoán private, viết `main.py`, chọn bài nộp cuối.
- Kết thúc: 6 tiếng sau giờ bắt đầu; bảng private được mở ngay sau đó.

## Rules

- Không dùng model âm thanh pretrained (wav2vec, Whisper, HuBERT, PANNs...). Được dùng weights ImageNet trong danh sách của BTC làm backbone cho spectrogram nếu muốn, nhưng thường train từ đầu một CNN nhỏ là đủ.
- Không dùng dữ liệu ngoài bộ được cấp (kể cả Speech Commands gốc). Được tự tạo dữ liệu tăng cường (trộn nhiễu, dịch thời gian, đổi tốc độ...).
- 5 tiếng đầu: tối đa 20 lần nộp public. Tiếng cuối: mật khẩu `private_test.zip` được công bố, tối đa 5 lần nộp private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Không mở `private_test.zip` trước khi mật khẩu được công bố.
- Tối đa 20 lần nộp; chọn bài nộp cuối để chấm private.

## Citation

Warden, P. (2018). Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition. arXiv:1804.03209. Bản Hugging Face mteb/speech-commands-mini.

## Dataset Description

```
train/*.wav            11.823 đoạn âm thanh sạch của 1.050 người nói
train.csv              id, label, speaker (mã người nói, để chia validation theo người)
test/*.wav             2.281 đoạn public
private_test.zip       (có mật khẩu) private_test/*.wav: 2.109 đoạn private
sample_submission.csv  id,label (toàn unknown)
baseline.ipynb, baseline.py
LICENSE-DATASET.txt
```

- Test gồm 449 người nói khác hẳn train (public và private cũng khác người nói với nhau).
- Khoảng 70% file test đã bị trộn tạp âm (nhiễu trắng, nhiễu hồng, nhiễu nâu, hoặc tiếng nhiều người nói chồng lên) với SNR 0-20 dB, dịch thời gian tới 0,125 giây và đổi âm lượng. Train là âm thanh sạch: phải tự tạo dữ liệu tăng cường giống test.
- Một số file train ngắn hơn 1 giây: đệm 0 cho đủ 16.000 mẫu.

Giải nén private test (khi có mật khẩu):

```python
import zipfile
zipfile.ZipFile("private_test.zip").extractall(".", pwd=b"<mật khẩu>")   # hoặc: unzip -P <mật khẩu> private_test.zip
```

## Data license

CC BY 4.0

## Files để tải lên

- Data: `SpeechCommands_KWS_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py` (hàm `macro_f1`), cao hơn là tốt hơn
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Mật khẩu private test: `host/mat_khau_private_test.txt`, công bố ở tiếng cuối
- Kiểm tra bằng sandbox submission: nộp `host/nop_dung_het.csv` (nếu có) phải ra điểm tối đa; nộp `submission.csv` của baseline phải gần `host/min_score.txt` (public 0.3771, private 0.3461)
