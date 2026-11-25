# Dạng 8 - Âm thanh: nhận dạng từ khoá, phân loại âm thanh

## 1. Nhận dạng đề

- Đầu vào: file `.wav` ngắn (1-5 giây). Đầu ra: nhãn (lệnh nói, loại âm thanh môi trường, người nói, cảm xúc giọng nói...).
- Biến thể: test là **người nói mới**, có **tạp âm**, độ dài khác nhau, lớp "unknown / silence".
- Bài tự host: SpeechCommands_KWS (10 lệnh + unknown, người nói mới, 70% test có tạp âm).

## 2. Metric

Accuracy hoặc macro-F1 (lớp `unknown` thường rất đông, macro-F1 coi nó nặng bằng một lệnh).

## 3. Lộ trình

| Bước | Việc | Kỳ vọng (SpeechCommands_KWS) |
|---|---|---|
| 0 | Baseline BTC (log-mel thống kê + logistic), nộp | 0,377 / 0,346 |
| 1 | Log-mel spectrogram (64-80 dải, cửa sổ 25 ms, bước 10 ms), CNN 2D nhỏ train từ đầu | tăng mạnh |
| 2 | **Trộn tạp âm** khi train (SNR ngẫu nhiên 0-20 dB), dịch thời gian, SpecAugment; val có tạp âm giống test | tăng mạnh trên test có tạp âm |
| 3 | TTA dịch thời gian, ensemble 2-3 seed | |

## 4. Khung code

Đọc wav và log-mel bằng torch (không cần torchaudio):

```python
from scipy.io import wavfile
def doc_wav(p, L=16000):
    sr, a = wavfile.read(p); a = a.astype(np.float32) / 32768
    return np.pad(a, (0, max(0, L - len(a))))[:L]

def mel_fb(n_mel=64, n_fft=512, sr=16000):
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    b = np.floor((n_fft + 1) * hz(np.linspace(0, 2595 * np.log10(1 + sr / 2 / 700), n_mel + 2)) / sr).astype(int)
    fb = np.zeros((n_mel, n_fft // 2 + 1), np.float32)
    for i in range(n_mel):
        fb[i, b[i]:b[i + 1]] = (np.arange(b[i], b[i + 1]) - b[i]) / max(1, b[i + 1] - b[i])
        fb[i, b[i + 1]:b[i + 2]] = (b[i + 2] - np.arange(b[i + 1], b[i + 2])) / max(1, b[i + 2] - b[i + 1])
    return torch.tensor(fb)

FB, WIN = mel_fb().to(DEV), torch.hann_window(400, device=DEV)
def logmel(x):                                     # x: (B, 16000) trên GPU
    S = torch.stft(x, 512, hop_length=160, win_length=400, window=WIN, return_complex=True).abs() ** 2   # (B, 257, 101)
    return torch.log(FB @ S + 1e-6).unsqueeze(1)   # (B, 1, 64, 101)
```

Trộn nhiễu theo SNR và SpecAugment (trên GPU, mỗi batch):

```python
def tron_nhieu(x, nhieu, snr_db):                  # x, nhieu: (B, L); snr_db: (B,)
    px, pn = x.pow(2).mean(1, keepdim=True) + 1e-8, nhieu.pow(2).mean(1, keepdim=True) + 1e-8
    return x + nhieu * torch.sqrt(px / pn / 10 ** (snr_db[:, None] / 10))

def spec_aug(S, F_=10, T_=20):                     # S: (B, 1, M, T)
    B, _, M, T = S.shape
    f0 = torch.randint(0, M - F_, (B, 1, 1, 1), device=S.device); t0 = torch.randint(0, T - T_, (B, 1, 1, 1), device=S.device)
    m = torch.arange(M, device=S.device).view(1, 1, M, 1); t = torch.arange(T, device=S.device).view(1, 1, 1, T)
    che = ((m >= f0) & (m < f0 + torch.randint(0, F_, (B, 1, 1, 1), device=S.device))) | \
          ((t >= t0) & (t < t0 + torch.randint(0, T_, (B, 1, 1, 1), device=S.device)))
    return S.masked_fill(che, S.mean().item())
```

Nguồn nhiễu: nhiễu trắng `torch.randn`, nhiễu màu (lọc trong miền tần số), và **trộn 2-3 đoạn âm thanh train khác** (giống tiếng người nói nền).

CNN nhỏ:

```python
def khoi(ci, co): return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True),
                                       nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True), nn.MaxPool2d(2))
model = nn.Sequential(nn.BatchNorm2d(1), khoi(1, 32), khoi(32, 64), khoi(64, 128), khoi(128, 256),
                      nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, C))
```

Hoặc ResNet-18 ImageNet trên spectrogram (lặp kênh thành 3, resize 128 x 128) nếu luật cho dùng backbone ImageNet.

## 5. Nút vặn

1. Augmentation giống điều kiện test (tạp âm, âm lượng, dịch thời gian) quan trọng nhất.
2. Val chia theo **người nói** (`GroupKFold`).
3. Chuẩn hoá spectrogram (BatchNorm đầu vào, hoặc trừ trung bình theo dải).
4. Lớp `unknown` lớn: trọng số lớp, chỉnh hệ số trên OOF cho macro-F1.
5. TTA: dịch ±100 ms, trung bình xác suất.

## 6. Bẫy

- File ngắn hơn 1 giây, stereo, sample rate khác 16 kHz, int16 hay float32: kiểm tra `wavfile.read` trả gì, chuẩn hoá về float mono 16 kHz.
- Đọc hàng chục nghìn file nhỏ chậm (ổ mạng / HDD): đọc một lần, lưu mảng `.npy` (N, 16000).
- Lật / đảo thời gian là augmentation sai cho tiếng nói.

## 7. Bài luyện

- Code tay: [d01_am_thanh](../../luyen_tap/CV/d01_am_thanh/) (log-mel tự viết, CNN, trộn nhiễu, SpecAugment).
- Thi thử: [SpeechCommands_KWS](../../selfhost_kaggle/SpeechCommands_KWS/).

## 8. Đọc thêm

- Warden, *Speech Commands* ([arXiv:1804.03209](https://arxiv.org/abs/1804.03209)).
- Park và cộng sự, *SpecAugment* ([arXiv:1904.08779](https://arxiv.org/abs/1904.08779)).
- Haytham Fayek, [Speech Processing for Machine Learning: Filter banks, MFCCs](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html).
