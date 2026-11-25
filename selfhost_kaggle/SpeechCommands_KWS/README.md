# Nhận dạng từ khoá trong âm thanh (Speech Commands)

Dạng bài **âm thanh**: nghe đoạn ghi âm 1 giây và cho biết người nói đang nói lệnh nào trong 10 lệnh điều khiển, hay một từ khác. Test là **người nói chưa gặp** khi train và phần lớn bị **lẫn tạp âm**.

## Overview

Mỗi file là một đoạn `.wav` 16 kHz, 16 bit, một kênh, dài tối đa 1 giây. Có 11 lớp:

- 10 lệnh: `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`;
- `unknown`: một trong 25 từ khác (`bed`, `bird`, `cat`, `happy`, `zero`... `nine`, `forward`, `backward`...), chiếm khoảng một nửa dữ liệu.

**Nhiệm vụ**: với mỗi đoạn âm thanh test, dự đoán một trong 11 nhãn.

**Luật (giống vòng khu vực OlpAI'26)**:

- Không dùng model âm thanh pretrained (wav2vec, Whisper, HuBERT, PANNs...). Được dùng weights ImageNet trong danh sách của BTC làm backbone cho spectrogram nếu muốn, nhưng thường train từ đầu một CNN nhỏ là đủ.
- Không dùng dữ liệu ngoài bộ được cấp (kể cả Speech Commands gốc). Được tự tạo dữ liệu tăng cường (trộn nhiễu, dịch thời gian, đổi tốc độ...).
- 5 tiếng đầu: tối đa 20 lần nộp public. Tiếng cuối: mật khẩu `private_test.zip` được công bố, tối đa 5 lần nộp private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.

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

## Data

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

## Hướng làm gợi ý

- Log-mel spectrogram (`torch.stft` hoặc `torchaudio` nếu có), CNN 2D nhỏ train từ đầu.
- Tăng cường: trộn nhiễu ở nhiều mức SNR, dịch thời gian, SpecAugment (che dải tần, che đoạn thời gian).
- Validation theo người nói (`GroupKFold` theo `speaker`), và nên có một bản validation đã trộn nhiễu để giống test.
- Lớp `unknown` lớn: dùng trọng số lớp hoặc chỉnh ngưỡng để tối ưu macro-F1.

Nguồn: Speech Commands v0.02 (Warden, 2018), bản Hugging Face `mteb/speech-commands-mini`, giấy phép CC BY 4.0.
