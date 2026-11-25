# Bài 01 (1/12) - Âm thanh: log-mel tự tính, CNN, trộn tạp âm, SpecAugment

**Thời gian mục tiêu: ≤ 50 phút.** Lời giải mẫu chạy 2,5 phút trên GTX 1650.

## Đọc trước khi làm

Đọc trong khoảng 30-45 phút trước khi bấm giờ; ghi ra giấy 3-5 ý chính. Lúc làm bài thì đóng hết.

- **Bài học**: [on_tap/dang_bai/08_am_thanh.md](../../../on_tap/dang_bai/08_am_thanh.md).
- **PyTorch**: [torch.stft](https://pytorch.org/docs/stable/generated/torch.stft.html) (`n_fft`, `hop_length`, `win_length`, `window`, `return_complex=True`), [torch.hann_window](https://pytorch.org/docs/stable/generated/torch.hann_window.html).
- **SciPy**: [scipy.io.wavfile.read](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html) (trả int16, phải chia 32768).
- **Lý thuyết**: Haytham Fayek, [Speech Processing for Machine Learning](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html) (khung 25 ms, bước 10 ms, bộ lọc mel tam giác, log).
- **Bài báo**: Park và cộng sự, *SpecAugment* ([arXiv:1904.08779](https://arxiv.org/abs/1904.08779)), mục 2 (che dải tần, che đoạn thời gian).

## Đề

- **Đầu vào**: `data/train/*.wav` + `data/train.csv` (`id,label,speaker`; 1.800 đoạn **sạch** của 40 người nói), `data/test/*.wav` + `data/test.csv` (`id`; 900 đoạn của 20 người nói **khác**). File wav 16 kHz, 1 giây, int16.
- **Lớp** (9): 8 "từ khoá" `len, xuong, trai, phai, dung, di, bat, tat` và `khac` (tổ hợp âm tiết khác, khoảng 1/3 dữ liệu). Mỗi từ khoá là chuỗi 1-3 âm tiết, khác nhau ở nguyên âm (formant) và đường cao độ (bằng / lên / xuống / gãy) **tương đối** so với giọng người nói.
- **Đầu ra**: `submission.csv` (`id,label`).
- **Metric**: macro-F1. **Ngưỡng 0.85** (lời giải mẫu 0.925).
- **Bẫy**: 80% đoạn test bị trộn tạp âm (nhiễu trắng, nhiễu hồng, tiếng nói khác chồng lên) với SNR 0-15 dB, lệch thời gian, đổi âm lượng. Cùng lời giải nhưng **không trộn tạp âm khi train chỉ được 0.35**.

## Khung cần viết (`bai_lam.py`)

1. Đọc wav, đệm / cắt đủ 16.000 mẫu, nạp hết lên GPU (N, 16000).
2. Log-mel tự tính: `torch.stft` (n_fft 512, hop 160, win 400, cửa sổ Hann), bộ lọc mel tam giác 64 dải, `log(... + 1e-6)`.
3. Augmentation mỗi batch, trên GPU: dịch thời gian ±0,1 s; trộn nhiễu (trắng / hồng / tổng 3 đoạn train khác) theo SNR ngẫu nhiên; đổi âm lượng; SpecAugment.
4. CNN 2D nhỏ (3-4 khối conv-BN-ReLU-pool, BatchNorm ở đầu vào), AdamW + OneCycle, label smoothing.
5. Val theo **người nói** (`speaker`), và làm nhiễu val giống test.
6. Bản cuối train trên toàn bộ train; dự đoán theo batch trong `inference_mode`.

## Bài 2 - Viết lại nhanh bài 30 (≤ 30 phút)

Trong `d30_du_lieu_bang`: `viet_lai.py`, chấm `python check_30.py viet_lai.py`.

## Cách chạy

```
python tao_du_lieu.py
python check_01.py bai_lam.py
```

## Mở rộng (không chấm)

- TTA dịch thời gian (±50, ±100 ms), trung bình xác suất.
- Thay CNN bằng ResNet-18 (`weights=None`, `conv1` nhận 1 kênh) và so điểm / thời gian.
- Chạy cùng khung trên [selfhost_kaggle/SpeechCommands_KWS](../../../selfhost_kaggle/SpeechCommands_KWS/) (tiếng nói thật, 11 lớp).
